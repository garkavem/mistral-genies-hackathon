import streamlit as st
import pandas as pd
from evaluator import MistralEvaluator
from chatbot_evaluator import *
import altair as alt
import os

def main():
    st.set_page_config(
        page_title="💡 Genies - LLM Quality Analysis ",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded")
    os.environ['MISTRAL_API_KEY'] = st.sidebar.text_input('Mistral API Key', type='password')
    alt.themes.enable("dark")

    st.markdown("# Genies - LLM Quality Analysis ")
    st.markdown("### 📝 Project Description:")
    project_description = st.text_input(" Please provide a brief description of your LLM Based application:")

    # Toggle for user's dialogs
    on = st.toggle("📂 I have my own tests")
    if on:
        uploaded_file = st.file_uploader("📤 Upload your dataset(CSV)", key="file_uploader")
        # If a file is uploaded, read and display its contents
        if uploaded_file is not None:
            # Read the CSV file into a DataFrame
            combined_dialogues = pd.read_csv(uploaded_file, index_col=0)

            # Evaluate the chatbot
            mistral_judge = MistralJudge(project_description)
            results = mistral_judge.evaluate(combined_dialogues)

            # Display the results in Streamlit
            st.markdown("### Evaluation Results")
            col = st.columns((4.5, 4.5), gap='medium')
            example_idx = -1
            with col[0]:
                st.markdown('#### Example')
                example = '###**Dialogue**###\n\n' + results.iloc[example_idx]['Dialogue'].replace(
                    '\n', '\n\n').replace('Human:', '**Human:**').replace(
                    'Travel Assistant:', '**Travel Assistant:**') + '\n\n###**Comment**###'
                score = results.iloc[example_idx]['Overall Satisfaction']
                comment = results.iloc[example_idx]['Comment']
                example += f'**[{score}/5]** {comment}\n\n'
                st.write(example)
            with col[1]:
                st.markdown('#### Evaluation Scores on the dataset')
                st.metric(label="Overall Satisfaction", value=round(results['Overall Satisfaction'].mean(), 1))
                st.markdown("""---""")
                st.metric(label="Understanding User Queries",
                          value=round(results['Understanding User Queries'].mean(), 1))
                st.metric(label="Providing Relevant Information",
                          value=round(results['Providing Relevant Information'].mean(), 1))
                st.metric(label="User Interaction and Engagement",
                          value=round(results['User Interaction and Engagement'].mean(), 1))
                st.metric(label="Personalization and Context Awareness",
                          value=round(results['Personalization and Context Awareness'].mean(), 1))
    else:
        mistral_evaluator = MistralEvaluator(project_description)
        st.markdown("### 📈 Metrics")
        # todo : replace options by generated metrics
        # generated_metrics = mistral_evaluator.generate_metrics()
        predefined_metrics = [
            "Grammatical Accuracy",
            "Toxicity",
            "Hallucination",
            "Coherence",
            "Personalization",
            "Sensitive Information",
        ]

        selected_metrics = st.multiselect("Select metrics:", predefined_metrics)
        # Allow user to add custom metrics
        custom_metric = st.text_input("You can add a custom metric:")
        if custom_metric and custom_metric not in selected_metrics:
            selected_metrics.append(custom_metric)

        # Allow user to specify the number of generated tests
        num_tests = st.number_input("Specify the number of generated user inputs:", min_value=1, max_value=100,
                                    value=3)

        if st.button("✔️ Confirm Selection"):
            if selected_metrics:
                st.markdown("### 📊 Generated Dataset")
                st.markdown("Based on the selected metrics, here are some examples of generated user inputs")
                generated_questions = mistral_evaluator.generate_questions(selected_metrics, num_tests)
                st.markdown(f"{generated_questions}")


if __name__ == "__main__":
    main()
