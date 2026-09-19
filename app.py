"""Local chat interface for the existing Materials-Informatics Copilot."""

import streamlit as st

TOOL_LABELS = {
    "get_material": "Materials Project",
    "search_papers": "Literature search",
}


def get_reply(question: str) -> dict:
    """Keep only the final answer and safe tool labels, never internal traces."""
    trace = []
    try:
        from copilot import ask_material_question

        answer = ask_material_question(question, tool_trace=trace)
        labels = list(dict.fromkeys(
            TOOL_LABELS[call["name"]] for call in trace
            if call.get("name") in TOOL_LABELS
        ))
        return {"role": "assistant", "content": answer, "tools": labels, "error": False}
    except Exception:
        # Do not render exception text: provider errors can contain request details.
        return {
            "role": "assistant",
            "content": "I couldn't complete this request. Please try again shortly. "
                       "If the issue persists, check your local API configuration and literature index.",
            "tools": [], "error": True,
        }


def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("tools"):
            with st.expander("Tools used"):
                for label in message["tools"]:
                    st.write(label)


def main() -> None:
    st.set_page_config(page_title="Materials-Informatics Copilot", page_icon="🔬", layout="centered")
    st.title("Materials-Informatics Copilot")
    st.write("Materials Project computed properties, combined with retrieval over a curated HEA arXiv abstract corpus.")
    st.caption("Computed values are structure-specific, not necessarily experimental. Literature evidence comes from abstracts, not full papers.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Each question is answered independently. Include the material or alloy name in each question.")
    if not st.session_state.messages:
        st.info("Ask about a computed material property or evidence from HEA abstracts. For example: “What is the band gap of TiO2?”")
    for message in st.session_state.messages:
        show_message(message)

    question = st.chat_input("Ask a materials question")
    if question and question.strip():
        user_message = {"role": "user", "content": question.strip()}
        st.session_state.messages.append(user_message)
        show_message(user_message)
        with st.spinner("Checking materials data and literature as needed. This may take a minute…"):
            reply = get_reply(question.strip())
        st.session_state.messages.append(reply)
        st.rerun()


if __name__ == "__main__":
    main()
