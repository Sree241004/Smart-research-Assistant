📚 Smart Research Assistant
An AI-powered research companion built with Streamlit and Hugging Face Transformers.
This tool helps students, teachers, and startup teams quickly extract insights from documents and stay updated with fresh online content.

🚀 Features
-> Upload Documents (PDF or TXT) → Auto summary with key insights
-> Ask Questions → Get evidence-based answers with confidence scores
-> Challenge Mode → Test your comprehension with AI-generated questions
-> Live Feed Integration → Summarize news/blog updates in real-time
-> Usage Dashboard → Track reports, questions, answers, and credits used

🛠️ Tech Stack
Streamlit
 – Frontend & UI
Hugging Face Transformers
 – NLP models (Summarization, QA, Text Generation)
pdfminer.six
 – PDF text extraction
Pandas
 – Feed and usage tracking

📦 Installation
Clone the repo

git clone https://github.com/Sree241004/Smart-research-Assistant.git
cd Smart-research-Assistant


Create a virtual environment

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt

▶️ Usage
Run the app with:
streamlit run app.py


Then open the local URL shown in the terminal (usually http://localhost:8501).

📊 Example Workflow
-> Upload a PDF (e.g., research paper, report, article)
-> View the summary & key takeaways
-> Ask natural-language questions about the content
-> Try the Challenge Me mode to test your knowledge
-> Explore the Live Feed for updated insights

📂 Project Structure
Smart-research-Assistant/
│── app.py                 # Main Streamlit app
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── /data                  # (Optional) Sample documents
│── /venv                  # Virtual environment (ignored by git)

💳 Billing & Usage (Demo)
Reports generated → deduct credits
Questions asked → deduct credits
Answers saved → deduct credits
Visible counters are shown in the sidebar

🌱 Future Improvements
-> Integration with live APIs (news/blogs, academic sources)
-> Multi-file document processing
-> Better citation handling for trustable insights
-> Export reports to PDF/Word
