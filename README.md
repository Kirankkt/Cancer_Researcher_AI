The issue: Right now when the code is run in streamlit the chatbot is not able to understand the user queries related to the previous response...the conversation history seems to be not working even though it is working in colab!.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Cancer Chatbot

## Overview
The Cancer Chatbot is an interactive application designed to provide users with information and support related to cancer. It utilizes natural language processing and machine learning techniques to answer questions, provide resources, and assist users in understanding cancer-related topics.

## Features
- **Natural Language Understanding**: The chatbot can comprehend and respond to user inquiries about cancer types, treatments, symptoms, and support resources.
- **Interactive Conversations**: Users can engage in dynamic conversations, receiving tailored responses based on their questions.
- **Resource Links**: Provides links to reliable cancer-related resources and support organizations!.

## Technologies Used
- **Python**: The primary programming language used for developing the chatbot.
- **Streamlit**: For creating a user-friendly web interface.
- **LangChain**: For managing conversations and utilizing language models.
- **FAISS**: For efficient similarity search and retrieval of information from a knowledge base.

## Installation
To run the Cancer Chatbot locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Dorcatz123/cancer_chatbot.git
   cd cancer_chatbot
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Chatbot
To start the chatbot, run the following command:
```bash
streamlit run cancer_chatbot.py
```

After running the command, open your browser and go to `http://localhost:8501` to interact with the chatbot.

## Usage
- Ask the chatbot questions about cancer.
- Receive information on various cancer topics, including symptoms, treatment options, and support resources based on the current research findings.

## Contributing
Contributions are welcome! If you have suggestions for improvements or features, please create a pull request or open an issue.

## License
This project is licensed under the GPL License. See the [LICENSE](LICENSE.txt) file for details.

## Contact
For any inquiries or feedback, please contact [Akshay P R](mailto:akshaypr314159@gmail.com).

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


