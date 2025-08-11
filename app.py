import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import json
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Student Eligibility Checker",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .subject-header {
        color: #4169E1;
        font-size: 1.5rem;
        border-bottom: 2px solid #4169E1;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .question-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4169E1;
    }
    .mcq-option {
        margin-left: 1rem;
        color: #2F4F4F;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data Models
class MCQQuestion(BaseModel):
    question: str = Field(description="The question text")
    options: List[str] = Field(description="List of 4 multiple choice options")
    correct_answer: str = Field(description="The correct answer from the options")

class ShortQuestion(BaseModel):
    question: str = Field(description="The short answer question")
    expected_answer: str = Field(description="Expected answer or key points")

class SubjectQuestions(BaseModel):
    subject: str = Field(description="Subject name")
    mcq_questions: List[MCQQuestion] = Field(description="List of MCQ questions")
    short_questions: List[ShortQuestion] = Field(description="List of short answer questions")

# Initialize session state
if 'questions_generated' not in st.session_state:
    st.session_state.questions_generated = False
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = None

def initialize_langchain(api_key):
    """Initialize LangChain with OpenAI"""
    os.environ["OPENAI_API_KEY"] = api_key
    llm = OpenAI(temperature=0.7, max_tokens=2000)
    return llm

def get_syllabus_context(next_class: int, subject: str) -> str:
    """Get syllabus context for the given class and subject"""
    
    # Define past classes to consider
    if next_class <= 3:
        past_classes = list(range(1, next_class))
    else:
        past_classes = list(range(next_class-3, next_class))
    
    syllabus_context = {
        "Math": {
            1: "Numbers 1-100, Basic addition, subtraction, Shapes recognition",
            2: "Numbers 1-1000, Multiplication tables 1-5, Time reading, Money concepts",
            3: "Numbers 1-10000, Division, Fractions basics, Geometry shapes",
            4: "Large numbers, Long multiplication/division, Fractions operations, Area perimeter",
            5: "Decimals, Percentage basics, LCM HCF, Angles, Triangles",
            6: "Integers, Ratio proportion, Algebra basics, Circles, Mensuration",
            7: "Rational numbers, Linear equations, Triangles properties, Statistics basics",
            8: "Exponents, Factorization, Quadrilaterals, Probability, Graphs",
            9: "Real numbers, Polynomials, Coordinate geometry, Trigonometry basics",
            10: "Arithmetic progressions, Quadratic equations, Circles theorems, Surface area volume"
        },
        "Physics": {
            6: "Light shadows, Motion types, Magnetism basics, Electricity simple circuits",
            7: "Heat temperature, Motion graphs, Reflection refraction, Current electricity",
            8: "Force pressure, Sound waves, Light phenomena, Chemical effects current",
            9: "Motion laws, Gravitation, Work energy, Sound properties, Natural phenomena",
            10: "Light reflection refraction, Electricity domestic circuits, Magnetic fields, Energy sources"
        },
        "Chemistry": {
            6: "Materials classification, Changes around us, Air water, Garbage management",
            7: "Acids bases salts, Physical chemical changes, Weather climate, Waste water",
            8: "Synthetic fibers, Metals non-metals, Coal petroleum, Combustion flame",
            9: "Matter nature, Atoms molecules, Structure atom, Diversity living organisms",
            10: "Chemical reactions, Acids bases, Metals non-metals, Carbon compounds, Periodic classification"
        }
    }
    
    context = f"AP State Syllabus for {subject}:\n"
    for class_num in past_classes:
        if class_num in syllabus_context.get(subject, {}):
            context += f"Class {class_num}: {syllabus_context[subject][class_num]}\n"
    
    return context

def create_question_chain(llm, subject: str, next_class: int, num_questions: int):
    """Create LangChain for question generation"""
    
    syllabus_context = get_syllabus_context(next_class, subject)
    
    # Calculate MCQ and short question counts
    mcq_count = num_questions // 2
    short_count = num_questions - mcq_count
    
    prompt_template = PromptTemplate(
        input_variables=["subject", "syllabus", "mcq_count", "short_count", "next_class"],
        template="""
        You are an expert educator creating eligibility test questions for AP State syllabus.
        
        Subject: {subject}
        Student wants to join Class: {next_class}
        Syllabus Context: {syllabus}
        
        Generate exactly {mcq_count} Multiple Choice Questions (MCQs) and {short_count} Short Answer Questions.
        
        For MCQs:
        - Each question should have exactly 4 options (A, B, C, D)
        - Questions should test understanding of concepts from the syllabus context
        - Include the correct answer
        
        For Short Questions:
        - Questions should require 2-3 sentence answers
        - Focus on conceptual understanding and application
        - Include expected answer points
        
        Format your response as a JSON object with this structure:
        {{
            "subject": "{subject}",
            "mcq_questions": [
                {{
                    "question": "Question text here?",
                    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                    "correct_answer": "A) Option 1"
                }}
            ],
            "short_questions": [
                {{
                    "question": "Question text here?",
                    "expected_answer": "Expected answer points here"
                }}
            ]
        }}
        
        Make sure all questions are appropriate for students preparing for class {next_class}.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain

def generate_questions(llm, next_class: int, questions_per_subject: int):
    """Generate questions for all three subjects"""
    subjects = ["Math", "Physics", "Chemistry"]
    all_questions = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, subject in enumerate(subjects):
        status_text.text(f"Generating {subject} questions...")
        
        try:
            chain = create_question_chain(llm, subject, next_class, questions_per_subject)
            
            result = chain.run({
                "subject": subject,
                "syllabus": get_syllabus_context(next_class, subject),
                "mcq_count": questions_per_subject // 2,
                "short_count": questions_per_subject - (questions_per_subject // 2),
                "next_class": next_class
            })
            
            # Try to parse JSON response
            try:
                questions_data = json.loads(result)
                all_questions[subject] = questions_data
            except json.JSONDecodeError:
                # Fallback: create sample questions if JSON parsing fails
                st.warning(f"Could not parse {subject} questions. Using sample format.")
                all_questions[subject] = create_sample_questions(subject, questions_per_subject)
                
        except Exception as e:
            st.error(f"Error generating {subject} questions: {str(e)}")
            all_questions[subject] = create_sample_questions(subject, questions_per_subject)
        
        progress_bar.progress((i + 1) / len(subjects))
    
    status_text.text("Questions generated successfully!")
    return all_questions

def create_sample_questions(subject: str, num_questions: int):
    """Create sample questions as fallback"""
    mcq_count = num_questions // 2
    short_count = num_questions - mcq_count
    
    sample_questions = {
        "subject": subject,
        "mcq_questions": [],
        "short_questions": []
    }
    
    # Add sample MCQs
    for i in range(mcq_count):
        sample_questions["mcq_questions"].append({
            "question": f"Sample {subject} MCQ question {i+1}?",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correct_answer": "A) Option 1"
        })
    
    # Add sample short questions
    for i in range(short_count):
        sample_questions["short_questions"].append({
            "question": f"Sample {subject} short question {i+1}?",
            "expected_answer": "Sample expected answer points"
        })
    
    return sample_questions

def display_questions(questions_data):
    """Display generated questions in a formatted way"""
    for subject, data in questions_data.items():
        st.markdown(f'<div class="subject-header">📖 {subject}</div>', unsafe_allow_html=True)
        
        # Display MCQ Questions
        st.subheader("Multiple Choice Questions")
        for i, mcq in enumerate(data.get("mcq_questions", []), 1):
            with st.container():
                st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
                st.markdown(f"**Q{i}. {mcq['question']}**")
                
                for option in mcq.get("options", []):
                    st.markdown(f'<div class="mcq-option">{option}</div>', unsafe_allow_html=True)
                
                with st.expander("Show Correct Answer"):
                    st.success(f"✅ {mcq.get('correct_answer', 'Not specified')}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Short Answer Questions
        st.subheader("Short Answer Questions")
        mcq_count = len(data.get("mcq_questions", []))
        
        for i, short_q in enumerate(data.get("short_questions", []), mcq_count + 1):
            with st.container():
                st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
                st.markdown(f"**Q{i}. {short_q['question']}**")
                
                with st.expander("Show Expected Answer"):
                    st.info(f"📝 {short_q.get('expected_answer', 'Not specified')}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")

def export_questions(questions_data, next_class, questions_per_subject):
    """Export questions to downloadable formats"""
    
    # Create text format
    text_content = f"STUDENT ELIGIBILITY CHECKER - CLASS {next_class}\n"
    text_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    text_content += f"Questions per subject: {questions_per_subject}\n"
    text_content += "="*60 + "\n\n"
    
    for subject, data in questions_data.items():
        text_content += f"{subject.upper()}\n" + "-"*30 + "\n\n"
        
        # MCQ Questions
        text_content += "MULTIPLE CHOICE QUESTIONS:\n\n"
        for i, mcq in enumerate(data.get("mcq_questions", []), 1):
            text_content += f"Q{i}. {mcq['question']}\n"
            for option in mcq.get("options", []):
                text_content += f"    {option}\n"
            text_content += f"    Correct Answer: {mcq.get('correct_answer', 'Not specified')}\n\n"
        
        # Short Questions
        text_content += "SHORT ANSWER QUESTIONS:\n\n"
        mcq_count = len(data.get("mcq_questions", []))
        for i, short_q in enumerate(data.get("short_questions", []), mcq_count + 1):
            text_content += f"Q{i}. {short_q['question']}\n"
            text_content += f"    Expected Answer: {short_q.get('expected_answer', 'Not specified')}\n\n"
        
        text_content += "\n" + "="*60 + "\n\n"
    
    return text_content

# Main App Interface
def main():
    st.markdown('<h1 class="main-header">🎓 Student Eligibility Checker</h1>', unsafe_allow_html=True)
    st.markdown("**Generate customized question papers for AP State syllabus eligibility testing**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        api_key = st.text_input("🔑 OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key to generate questions")
        
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue")
            st.info("You can get your API key from: https://platform.openai.com/api-keys")
        
        st.markdown("---")
        
        # Input parameters
        st.subheader("📝 Test Parameters")
        next_class = st.selectbox(
            "Next Class to Join",
            options=list(range(2, 11)),
            index=4,  # Default to class 6
            help="Select the class the student wants to join"
        )
        
        questions_per_subject = st.slider(
            "Questions per Subject",
            min_value=2,
            max_value=20,
            value=10,
            step=2,
            help="Total questions per subject (split equally between MCQ and Short Answer)"
        )
        
        st.markdown("---")
        
        # Display calculation
        st.subheader("📊 Question Distribution")
        mcq_per_subject = questions_per_subject // 2
        short_per_subject = questions_per_subject - mcq_per_subject
        total_questions = questions_per_subject * 3
        
        st.metric("MCQ per Subject", mcq_per_subject)
        st.metric("Short Answer per Subject", short_per_subject)
        st.metric("Total Questions", total_questions)
        
        # Generate button
        generate_btn = st.button("🚀 Generate Questions", 
                                type="primary", 
                                disabled=not api_key,
                                help="Click to generate eligibility test questions")
    
    # Main content area
    if generate_btn and api_key:
        try:
            with st.spinner("Initializing AI system..."):
                llm = initialize_langchain(api_key)
            
            st.success("✅ AI system initialized successfully!")
            
            # Generate questions
            with st.spinner("Generating questions... This may take a few minutes."):
                questions = generate_questions(llm, next_class, questions_per_subject)
            
            st.session_state.questions_generated = True
            st.session_state.generated_questions = questions
            
            st.markdown('<div class="success-box">✅ <strong>Questions generated successfully!</strong> Scroll down to view the eligibility test.</div>', 
                       unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error generating questions: {str(e)}")
            st.info("Please check your API key and try again.")
    
    # Display generated questions
    if st.session_state.questions_generated and st.session_state.generated_questions:
        st.markdown("---")
        st.header("📋 Generated Eligibility Test")
        
        # Summary info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Target Class", next_class)
        with col2:
            st.metric("Questions/Subject", questions_per_subject)
        with col3:
            st.metric("Total Questions", questions_per_subject * 3)
        with col4:
            st.metric("Subjects", 3)
        
        st.markdown("---")
        
        # Display questions
        display_questions(st.session_state.generated_questions)
        
        # Export options
        st.header("📥 Download Test Paper")
        
        # Generate export content
        export_content = export_questions(
            st.session_state.generated_questions, 
            next_class, 
            questions_per_subject
        )
        
        # Download button
        st.download_button(
            label="💾 Download as Text File",
            data=export_content,
            file_name=f"eligibility_test_class_{next_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download the complete question paper as a text file"
        )
        
        # JSON download
        json_content = json.dumps(st.session_state.generated_questions, indent=2)
        st.download_button(
            label="📊 Download as JSON",
            data=json_content,
            file_name=f"eligibility_test_class_{next_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download the questions in JSON format for further processing"
        )
    
    elif not st.session_state.questions_generated:
        # Initial instructions
        st.info("""
        ### 📚 How to Use This Application:
        
        1. **Enter your OpenAI API Key** in the sidebar
        2. **Select the target class** the student wants to join
        3. **Choose the number of questions** per subject
        4. **Click Generate Questions** to create the eligibility test
        
        ### 📖 What This App Does:
        
        - Generates questions from **3 past classes** relevant to the target class
        - Creates **MCQ and Short Answer** questions for each subject
        - Follows **AP State syllabus** guidelines
        - Covers **Math, Physics, and Chemistry**
        - Provides **downloadable** question papers
        
        ### 🎯 Example:
        If target class is **6th**, questions will be generated from classes **3rd, 4th, and 5th**.
        """)

if __name__ == "__main__":
    main()