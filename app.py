import streamlit as st

from parser.profile_parser import extract_patient_profile
from agents.health_agent import HealthAgent
from utils.config import get_llm, APP_TITLE


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)


def main():

    st.title(APP_TITLE)

    st.write(
        "Personalized wellness coaching powered by Gemini, RAG, and Streamlit."
    )

    if "profile" not in st.session_state:
        st.session_state.profile = None

    st.header("Patient Onboarding")

    onboarding_text = st.text_area(
        "Paste patient onboarding notes",
        height=200
    )

    if st.button("Generate Profile"):

        llm = get_llm()

        profile = extract_patient_profile(
            onboarding_text,
            llm
        )

        st.session_state.profile = profile

        st.success(
            "Patient profile generated successfully."
        )

    if st.session_state.profile:

        st.subheader("Parsed Profile")

        st.json(
            st.session_state.profile.model_dump()
        )

    st.divider()

    st.header("Health Coach Chat")

    user_question = st.text_input(
        "Ask a question about the wellness protocol"
    )

    if user_question:

        agent = HealthAgent()

        answer, sources = agent.generate_response(
            user_question
        )

        st.write(answer)

        with st.expander(
            "View Sources"
        ):

            for i, doc in enumerate(
                sources,
                start=1
            ):

                st.markdown(
                    f"### Source {i}"
                )

                st.write(
                    doc.page_content
                )


if __name__ == "__main__":
    main()