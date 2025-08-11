# Student Eligibility Checker 📚

A Streamlit application that generates customized eligibility test questions for students wanting to join AP State curriculum classes.

## Features

- 🎯 Generate questions for **Math, Physics, and Chemistry**
- 📊 Creates both **MCQ and Short Answer** questions
- 🏫 Based on **AP State syllabus** (classes 3-10)
- 💾 **Download** question papers in text and JSON formats
- 🤖 Powered by **OpenAI GPT** for intelligent question generation

## How It Works

The app generates questions from **3 past classes** to test eligibility for the target class:
- Target Class 6 → Questions from Classes 3, 4, 5
- Target Class 9 → Questions from Classes 6, 7, 8

## Requirements

- Python 3.7+
- OpenAI API key
- Required packages (see Installation)

## Installation

1. Clone or download the application
2. Install required packages:
```bash
pip install streamlit openai langchain pydantic
```

## Usage

1. **Run the application:**
```bash
streamlit run app.py
```

2. **Configure in the sidebar:**
   - Enter your OpenAI API key
   - Select target class (2-10)
   - Choose questions per subject (2-20)

3. **Generate questions:**
   - Click "Generate Questions"
   - Wait for AI to create customized questions

4. **Download results:**
   - View questions in the web interface
   - Download as text file or JSON

## Example Output

For each subject, you get:
- **MCQ Questions** with 4 options and correct answers
- **Short Answer Questions** with expected answer points
- Questions covering relevant syllabus topics

## Subjects Covered

- **Mathematics:** Numbers, algebra, geometry, statistics
- **Physics:** Motion, light, electricity, sound, forces
- **Chemistry:** Matter, atoms, reactions, acids/bases

## API Key Setup

Get your OpenAI API key from: https://platform.openai.com/api-keys

## File Structure

```
├── app.py              # Main Streamlit application
├── README.md          # This file
└── requirements.txt   # Python dependencies (optional)
```

## Customization

You can modify:
- Syllabus content in `get_syllabus_context()` function
- Question templates in `create_question_chain()` function
- Styling in the CSS section

## Troubleshooting

**Common Issues:**
- **"Please enter API key"** → Add valid OpenAI API key
- **Question generation fails** → Check API key and internet connection
- **JSON parsing errors** → App will use fallback sample questions

## License

Open source - feel free to modify and use for educational purposes.

## Support

For issues or questions, please check:
- OpenAI API documentation
- Streamlit documentation
- Make sure all required packages are installed