import json

import requests
import sseclient
import streamlit as st

st.title("Chat with AI Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("What would you like to know?"):
    # 1) Append the user’s message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Create a placeholder for the streaming assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # We'll build up three distinct sections:
        #   a) tool_call_text     (e.g. “Tool call: CITY_WEATHER with args: {...}”)
        #   b) tool_output_text   (e.g. “Tool output: {...}”)
        #   c) assistant_text     (e.g. “Agent: It’s currently sunny...”)
        tool_call_text = ""
        tool_output_text = ""
        assistant_text = ""
        # For accumulating the streaming arguments for the tool call:
        tool_name = None
        tool_args_buffer = ""

        try:
            # Send request to our FastAPI backend (which streams SSE)
            response = requests.post(
                "http://localhost:8080/chat",
                json={"query": prompt},
                stream=True
            )
            client = sseclient.SSEClient(response)

            for event in client.events():
                if not event.data:
                    continue

                # Parse each SSE event as JSON
                try:
                    message_data = json.loads(event.data)
                except json.JSONDecodeError:
                    continue

                # Only process if it's a "message" type
                if message_data.get("type") != "message":
                    continue

                evt_type = message_data.get("event_type", "")
                content = message_data.get("content", "") or ""
                name = message_data.get("name", "")
                additional_kwargs = message_data.get("additional_kwargs", {})

                # ----- 1) TOOL CALL -----
                if evt_type == "tool_call":
                    calls = additional_kwargs.get("tool_calls", [])
                    for call in calls:
                        if isinstance(call, dict):
                            func = call.get("function", {})
                            # If this chunk has a non-null tool name, capture it:
                            if func.get("name"):
                                tool_name = func["name"]
                                # Reset the arguments buffer for a new call:
                                tool_args_buffer = ""
                            # Grab whatever arguments piece is present in this chunk:
                            arg_piece = func.get("arguments", "")
                            if arg_piece:
                                tool_args_buffer += arg_piece

                            # Build the display text after updating buffer:
                            if tool_name:
                                if tool_args_buffer.strip():
                                    tool_call_text = (
                                        f"**Tool call:** {tool_name} with args: {tool_args_buffer}\n\n"
                                    )
                                else:
                                    tool_call_text = f"**Tool call:** {tool_name} (no args)\n\n"
                            else:
                                tool_call_text = "**Tool call:** <unknown> (no args)\n\n"

                            # Re‐render all three sections so far
                            full_text = tool_call_text + tool_output_text + assistant_text
                            message_placeholder.markdown(full_text + "▌")

                # ----- 2) TOOL RESPONSE -----
                elif evt_type == "tool_response" and name:
                    raw = content.strip()
                    try:
                        parsed = json.loads(raw)
                        pretty = json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        pretty = raw

                    tool_output_text = f"**Tool output:** {pretty}\n\n"
                    full_text = tool_call_text + tool_output_text + assistant_text
                    message_placeholder.markdown(full_text + "▌")

                # ----- 3) FINAL ASSISTANT RESPONSE -----
                elif evt_type == "assistant":
                    if content:
                        # Only prepend "Agent: " on the very first assistant chunk
                        if assistant_text == "":
                            assistant_text = f"**Agent:** {content}"
                        else:
                            assistant_text += content

                        full_text = tool_call_text + tool_output_text + assistant_text
                        message_placeholder.markdown(full_text + "▌")

            # Streaming is done → show final, full response
            full_text = tool_call_text + tool_output_text + assistant_text
            message_placeholder.markdown(full_text)

            # 4) Save assistant’s final answer into chat history
            st.session_state.messages.append({"role": "assistant", "content": full_text})

        except Exception as e:
            st.error(f"Error: {str(e)}")
            message_placeholder.markdown("Sorry, there was an error processing your request.")
