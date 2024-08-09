__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import datetime
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
import bs4

import regex as re
from langchain.callbacks import StreamingStdOutCallbackHandler
import logging
from PIL import ImageGrab, Image
from openai import OpenAI
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import pyaudio
import time
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import base64
import streamlit as st
import sys
from groq import Groq
from langchain_community.document_loaders import ArxivLoader, WebBaseLoader
import pytz





# Access API keys from st.secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Configure your clients using the secrets
genai.configure(api_key=GENAI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=3000)


def load_documents(data):
    # Load PDF documents
    pdf_loader = PyPDFDirectoryLoader(data)
    pdf_documents = pdf_loader.load()

    
    #Load web pages
    web_loader = WebBaseLoader(
        web_paths=("https://www.epicurious.com","https://www.bbcgoodfood.com"),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title", "post-content", "post-header")))
    )
    web_documents = web_loader.load()

    from langchain_community.document_loaders import WikipediaLoader
    wiki = WikipediaLoader(query="cookbook", load_max_docs=2).load()

    docs=pdf_documents+web_documents+ wiki
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    doc = text_splitter.split_documents(documents=docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vectorstore = Chroma.from_documents(documents=doc, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore



def get_conversational_chain():
    prompt_template = """
You are an AI-powered cooking assistant designed to help users with their cooking tasks. Your role is to provide step-by-step instructions for recipes, ingredient substitutions, cooking tips, problem-solving solutions, and motivational support in a friendly manner. Ensure the user feels confident and assisted throughout the cooking process.

Tasks:
1. Provide each recipe step and confirm completion before proceeding.
2. Suggest alternatives for missing ingredients.
3. Offer helpful tips.
4. Help troubleshoot cooking issues.
5. Set timers for cooking tasks.
6. Engage in encouraging and friendly conversation.
7. Ensure each step is completed before moving on.
8. Always be supportive and positive.
9. Keep instructions clear and easy to follow.
10. Keep the conversation engaging and natural.
11. Continuously check the user's progress and address any issues.

before proceeding the answer just verify  or confirm with question about what type of dish looking only if the user giving general dish name.
Format your response clearly:

**Name**: [Recipe Name]
- **Preparation time**: [Time in minutes]
- **Ingredients**: 
  - [Ingredient 1: Quantity]
  - [Ingredient 2: Quantity]
  - [Ingredient 3: Quantity]
  - ...
- **Instructions**: 
  1. [Step 1]
  2. [Step 2]
  3. [Step 3]
  - ...
- **Nutritional information**: 
  - Carbohydrates: [Amount in grams]
  - Protein: [Amount in grams]
  - Fat: [Amount in grams]
  - Sugar: [Amount in grams]

Ensure that each section is separated by a newline and clearly formatted.

Context:
{context}

User Question:
{question}

Assistant Response:
"""
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=3000)

    try:
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logging.error(f"Error loading QA chain: {e}")
        print(f"Error setting up the conversation chain. Please check the log for details.")
        return None


def user_input(user_question):
    try:
        greetings = ["hello", "hai", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if user_question.lower() in greetings:
            yield from get_greeting().split()
            return

        embeddings = OpenAIEmbeddings()
        db2 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        docs = db2.similarity_search(user_question)
        streaming_handler = StreamingStdOutCallbackHandler()

        chain = get_conversational_chain()
        response = chain.invoke(
            {"input_documents": docs, "question": user_question},
            callbacks=[streaming_handler],
            return_only_outputs=True
        )
        if not response or not response.get("output_text"):
            yield "Apologies, I am unable to find the answer. Can you please rephrase your question?"
        else:
            formatted_response = response["output_text"].strip()
            yield formatted_response
    except Exception as e:
        logging.error(f"Error in user_input function: {e}")
        yield f"Sorry, something went wrong. Please try again later. Error: {str(e)}"

 
def get_indian_time():
    india_tz = pytz.timezone('Asia/Kolkata')
    return datetime.datetime.now(india_tz)


def get_greeting():
    current_time = get_indian_time()
    current_hour = current_time.hour
    if current_hour < 12:
        return "Good morning! How can I help you with making breakfast?"
    elif 12 <= current_hour < 16:
        return "Good afternoon! How can I assist you today with lunch?"
    elif 16 <= current_hour < 18:
        return "Good evening! Are you planning to prepare some snacks?"
    else:
        return "Hey... How can I help you prepare for dinner?"


def format_response(response):
    response = response.replace(' - ', ': ').replace('•', '*')
    response = re.sub(r'(\d+)', r'\n\1.', response)
    response = re.sub(r'\n\s*\n', '\n', response)
    return response.strip()


# VOICE ASSISTANT 

#Define wake word
# wake_word = "chef"  

# num_cores = os.cpu_count()
# whisper_size = 'base'
# whisper_model = WhisperModel(
#     whisper_size,
#     device='cpu',
#     compute_type='int8',
#     cpu_threads=num_cores // 2,
#     num_workers=num_cores // 2
# )

# r = sr.Recognizer()
# source = sr.Microphone()
# sys_msg = ( 'you are a multimodal AI voice assistant. Your user may request assistance for cooking '
#     ' Generate the most useful and stop after first instruction and say once you completed this let me know and continue after user response '
#     'factual response possible, carefully considering all previous generated text in your response before '
#     'adding new tokens to the response.  just use the context if added. '
#     'Use all of the context of this conversation so your response is relevant to the conversation. Make '
#     'your responses clear and concise, avoiding any verbosity.')

# convo = [{'role': 'system', 'content': sys_msg}]



# def groq_prompt(prompt):
#     """
#     Generate a response using GROQ based on the user prompt and optional image context.
#     """
#     convo.append({'role': 'user', 'content': prompt})
#     chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
#     response = chat_completion.choices[0].message
#     convo.append(response)
#     return response.content




# def execute_voice_assistant():
#     # Display initial listening message
#     listening_text = st.empty()
#     listening_text.info("I'm **CHEFMATE** 🎙️ Listening...")
#     start_listening()

# def speak(text):
#     player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
#     stream_start = False

#     with openai_client.audio.speech.with_streaming_response.create(
#             model='tts-1',
#             voice='onyx',
#             response_format='pcm',
#             input=text,
#     ) as response:
#         silence_threshold = 0.01
#         for chunk in response.iter_bytes(chunk_size=1024):
#             if stream_start:
#                 player_stream.write(chunk)
#             else:
#                 if max(chunk) > silence_threshold:
#                     player_stream.write(chunk)
#                     stream_start = True

# def wav_to_text(audio_path):
#     segments, _ = whisper_model.transcribe(audio_path)
#     text = ''.join(segment.text for segment in segments)
#     return text

# def callback(recognizer, audio):
#     prompt_audio_path = 'prompt.wav'
#     with open(prompt_audio_path, 'wb') as f:
#         f.write(audio.get_wav_data())

#     prompt_text = wav_to_text(prompt_audio_path)
#     clean_prompt = extract_prompt(prompt_text, wake_word)

#     if clean_prompt:
#         print(f'USER: {clean_prompt}')
#         response = groq_prompt(prompt=clean_prompt)

#         # response = ' '.join(user_input(clean_prompt))
#         print(f'ASSISTANT: {response}')
#         speak(response)


# def start_listening():
#     with source as s:
#         r.adjust_for_ambient_noise(s, duration=2)
#     st.write(f"\nSay '{wake_word}' followed by your prompt. \n")
#     r.listen_in_background(source, callback)

#     while True:
#         time.sleep(0.5)

# def extract_prompt(transcribed_text, wake_word):
#     pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
#     match = re.search(pattern, transcribed_text, re.IGNORECASE)

#     if match:
#         prompt = match.group(1).strip()
#         return prompt
#     else:
#         return None



# Image Analysis

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image



def get_gemini_response(input, image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    if not input and not image:
        return "Please provide the input text and image,then i can help you."
    
    if input and not image:
        return "Please provide the image also that you had."
    elif image and not input.strip():
        response = model.generate_content(image)
    else:
        response = model.generate_content([input, image])
    return response.text



def gemini_repsonse(prompt,image,input):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([input,image[0],prompt])
    return response.text  



def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")



# DISPLAYS 

def display_header(encoded_image):
    st.markdown(
        f"""
        <style>
            .header {{
                text-align: center;
                margin-top: 10px;
            }}
            .header img {{
                width: 300px;
            }}
        </style>
        <div class="header">
            <img src="data:image/jpeg;base64,{encoded_image}" alt="Chef Mate Logo"/>
        </div>
        """,
        unsafe_allow_html=True
    )



import random
def display_recipe_tip():
    recipetips = [
    "Read the entire recipe first to understand the steps and ingredients required.",
    "Prepare ingredients in advance by measuring and prepping them before cooking.",
    "Use fresh ingredients whenever possible for better flavor and nutritional value.",
    "Season as you go to build layers of flavor throughout the cooking process.",
    "Taste and adjust your dish continuously to ensure balanced seasoning, acidity, and sweetness.",
    "Control the heat appropriately; use high heat for searing and browning, and low heat for simmering and slow cooking.",
    "Don't overcrowd the pan; give your ingredients enough space to cook evenly and develop proper textures.",
    "Let meat rest after cooking to allow the juices to redistribute, resulting in a more tender and flavorful dish.",
    "Invest in good quality kitchen tools and equipment for more efficient and enjoyable cooking.",
    "Clean as you go to maintain an organized workspace and make the cooking process less stressful.",
    "Use a meat thermometer to ensure your meats are cooked to the correct temperature.",
    "Experiment with herbs and spices to enhance and diversify the flavors of your dishes.",
    "Always use the correct type and size of pan or baking dish as specified in the recipe.",
    "Don’t be afraid to make substitutions or adjustments based on your taste preferences or dietary needs.",
    "Allow baked goods to cool completely before cutting to avoid crumbling and ensure proper texture.",
    "Soak wooden skewers in water before grilling to prevent them from burning.",
    "Keep a well-stocked pantry with essential ingredients like oils, spices, and canned goods for quick meal preparation.",
    "Use a sharp knife for more efficient and safer cutting and chopping.",
    "Plan your meals ahead of time to reduce stress and ensure you have all necessary ingredients.",
    "Store leftovers properly in airtight containers to maintain freshness and prevent spoilage."
]

    recipe_tip = random.choice(recipetips)
    
    st.markdown(
    f"""
    <style>
        .marquee {{
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            box-sizing: border-box;
            animation: marquee 15s linear infinite; 
        }}
        @keyframes marquee {{
            0%   {{ transform: translate(100%, 0); }}
            100% {{ transform: translate(-100%, 0); }}
        }}
        .recipe-tip {{
            position: fixed;
            bottom: 0;
            width: 100%;
            text-color: red;
            padding: 10px;
            text-align: center;
            font-size: 18px;
            border-top: 1px solid #ddd;
        }}
    </style>
    <div class="recipe-tip">
        <div class="marquee">
            <strong>TIP:</strong> {recipe_tip}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)





def display_faq():
    st.title("FAQs")

    faq = {
    "What is Chef Mate?": "Chef Mate is an AI-powered cooking assistant that helps you find recipes, provides step-by-step cooking guidance, and offers nutritional information.",
    "How do I get recipe recommendations?": "Simply fill out the form with your meal preferences, maximum preparation time, ingredients to include, ingredients to exclude, and a description of the food you're into. Then, click 'Generate recommendation' to get a list of recipes.",
    "What kind of meals can Chef Mate recommend?": "Chef Mate can recommend recipes for Breakfast, Lunch, Dinner, Dessert, Snack, and Drinks.",
    "Can I exclude certain ingredients from the recipes?": "Yes, you can specify ingredients to exclude, and Chef Mate will ensure the recommended recipes do not contain those ingredients.",
    "Does Chef Mate provide nutritional information for the recipes?": "Yes, each recommended recipe includes nutritional information such as carbohydrates, protein, fat, and sugar content.",
    "Is there a limit to the number of recipes I can get at once?": "Chef Mate provides a list of 5 recipes that best match your preferences.",
    "Can Chef Mate help with dietary restrictions?": "Yes, by specifying ingredients to exclude, you can tailor the recommendations to fit dietary restrictions or preferences.",
    "How do I start using Chef Mate?": "Select a feature from the sidebar, fill out the required information, and click 'Generate recommendation' to get started.",
    "What if no recipes match my preferences?": "If no recipes are found, try adjusting your preferences and generating recommendations again.",
    "Can Chef Mate help with cooking instructions?": "Yes, each recipe comes with step-by-step cooking instructions to guide you through the preparation.",
    "Can I get recipes for specific cuisines?": "Currently, Chef Mate focuses on general recipe recommendations based on your ingredients and preferences. Future updates may include specific cuisine options.",
    "How accurate are the nutritional information provided?": "The nutritional information is estimated based on standard ingredient values. For precise dietary needs, please consult a nutritionist.",
    "Can Chef Mate handle multiple dietary restrictions?": "Yes, you can list multiple ingredients to exclude to accommodate various dietary restrictions.",
    "Is Chef Mate available on mobile devices?": "Chef Mate is a web-based application and can be accessed on mobile devices through a web browser.",
    "Do I need to create an account to use Chef Mate?": "Currently, Chef Mate does not require an account. You can start using it right away by filling out the form.",
    "How often are new recipes added to Chef Mate?": "Chef Mate's database is regularly updated to include new and diverse recipes.",
    "Can Chef Mate suggest recipes based on leftover ingredients?": "Yes, you can input any ingredients you have on hand, and Chef Mate will recommend recipes that use those ingredients.",
    "What if I have allergies?": "Make sure to list any allergens in the ingredients to exclude section to avoid recipes containing those allergens.",
    "Can I save my favorite recipes?": "Currently, Chef Mate does not support saving recipes, but you can bookmark the page in your browser for quick access."
}


    for question, answer in faq.items():
        with st.expander(question):
            st.write(f"**Answer**: {answer}")


# HOME TAB 

def home():
    """ Home Tab """
    st.title("🍳 Welcome to Chef Mate!")
    st.info("🔝⬅️ To navigate the features, click on the arrow in the top-left corner and select the desired feature.")

    st.write("""
    #### What is Chef Mate?

    **Chef Mate** is an AI-powered cooking assistant designed to enhance your culinary experience. With features like voice-assisted cooking guidance, chat interactions for recipe inquiries, ingredient-based recipe suggestions, and image analysis for cooking assistance, Chef Mate aims to make cooking enjoyable and hassle-free.

    #### How It Work?

    **Chef Mate** is a revolutionary generative AI tool that is capable of understanding your request and generating response leverages advanced technologies including GPT-4,Llama-3.1, FastWhisper, TTS (Text-to-Speech), Gemini, LangChain, RAG (Retrieval-Augmented Generation), and Chroma vector DB to provide personalized cooking assistance. By analyzing various cooking instructions and ingredients, the platform offers comprehensive and interactive cooking support.

    #### Features

    * **Voice Interaction:** Get step-by-step cooking guidance through voice commands.
    * **Chat Interaction:** Ask questions about recipes and get instant answers.
    * **Recipe Recommendations:** Enter the ingredients you have, and Chef Mate will suggest recipes you can make.
    * **Dish Identification:** Upload images of dishes or ingredients, and Chef Mate will help identify them and provide relevant cooking advice and recipes.
    * **Nutritional Analysis:** Upload images of ingredients or dishes to get detailed nutritional information, including calorie count and other nutritional details, and track total calorie intake.
    * **Frequently Asked Questions:** Get answers to common cooking-related questions and concerns.

    #### Upcoming Features

    Chef Mate is continuously evolving to offer more exciting features. Stay tuned for:
    - Enhanced personalized recipe vedio recommendations based on dietary preferences.
    - Integration with smart kitchen appliances for an even smoother cooking experience.
    - Multilanguage assistance!
    """)





#FOOD RECOMMENDATION

def fix_prep_time(obj):
    if isinstance(obj, datetime.datetime):
        return f"{obj.hour}:{obj.minute}"
    else:
        return obj

# Initialize the OpenAI model


    
# def build_prompt(food_category, preparation_time, included_ingredients, excluded_ingredients, description):
#     return f"""
#     You are a food advisor. Your task is to recommend recipes based on the following user preferences:

#     Food category: {food_category}
#     Maximum preparation time: {preparation_time}
#     Ingredients to include: {included_ingredients}
#     Ingredients to exclude: {excluded_ingredients}
#     Description: {description}

#     Provide a list of 5 recipes that best match these preferences. For each recipe, include:
#     - Name
#     - Preparation time
#     - Ingredients (with quantities)
#     - Instructions
#     - Nutritional information (carbohydrates, protein, fat, sugar)

#     Format your response as follows:
#     1. **Name**: [Recipe Name]
#        - **Preparation time**: [Time]
#        - **Ingredients**: [List of Ingredients with quantities]
#        - **Instructions**: [Steps]
#        - **Nutritional information**: Carbohydrates - [Amount], Protein - [Amount], Fat - [Amount], Sugar - [Amount]
#     """

def build_prompt(food_category, preparation_time, included_ingredients, excluded_ingredients, description):
    return f"""
You are an expert food advisor. Your goal is to provide personalized recipe recommendations based on the user's preferences.

User Preferences:
- **Food category**: {food_category}
- **Maximum preparation time**: {preparation_time} minutes
- **Ingredients to include**: {', '.join(included_ingredients)}
- **Ingredients to exclude**: {', '.join(excluded_ingredients)}
- **Additional description**: {description}

Please provide a list of 5 recipes that best match these preferences. For each recipe, include the following details:
1. **Name**: [Recipe Name]
2. **Preparation time**: [Time in minutes]
3. **Ingredients**: [List of ingredients with quantities]
4. **Instructions**: [Step-by-step preparation instructions]
5. **Nutritional information**: 
   - Carbohydrates: [Amount in grams]
   - Protein: [Amount in grams]
   - Fat: [Amount in grams]
   - Sugar: [Amount in grams]

Format the response as follows:
1. **Name**: [Recipe Name]
   - **Preparation time**: [Time in minutes]
   - **Ingredients**: 
     - [Ingredient 1: Quantity]
     - [Ingredient 2: Quantity]
     - [Ingredient 3: Quantity]
     - ...
   - **Instructions**: 
     1. [Step 1]
     2. [Step 2]
     3. [Step 3]
     ...
   - **Nutritional information**: 
     - Carbohydrates: [Amount in grams]
     - Protein: [Amount in grams]
     - Fat: [Amount in grams]
     - Sugar: [Amount in grams]

Ensure the recipes are clear, easy to follow, and suitable for the user's preferences.
    """


# FEATURES

def main():
    st.set_page_config(page_title="ChefMate", page_icon="🍳",layout="wide")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    feature = st.sidebar.radio(
    "Features",
    [
        ":rainbow[**Home**]",
        "**Voice Assistant**",
        "**Chat Interaction**",
        "**Recipe Recommendations**",
        "**Nutritional Analyst**",
        "**Dish Identification**",
        "**Frequently Asked Questions**"
    ],
    captions=[
        "",
        "Get step-by-step cooking guidance through voice commands.",
        "Ask questions about recipes and get instant answers.",
        "Enter the ingredients,time,meal etc you have, and Chef Mate will suggest recipes you can make.",
        "Upload images of ingredients or dishes to get detailed nutritional information, including calorie count and other nutritional details, and track total calorie intake.",
        "Upload images of dishes or ingredients, and Chef Mate will help identify them and provide relevant cooking advice.",
        "Get answers to common cooking-related questions and concerns."
    ]
)

    
    

    if feature == ":rainbow[**Home**]":
        image_path = r"assests/chef.png"
        encoded_image = load_image(image_path)
        display_header(encoded_image)
        home()
        display_recipe_tip()

    

    elif feature == "**Voice Assistant**":
       
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        with col3:
            image_path = "assests/achef.png" 
            encoded_image = load_image(image_path)
            display_header(encoded_image)

            # Center the button below the image
            st.markdown(
                """
                <style>
                .center-button .stButton button {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 24px;
                    text-align: center;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border: none;
                    border-radius: 12px;
                }
                .center-button .stButton button:hover {
                    background-color: #45a049;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
           
               
            # with col3:
            #     st.markdown('<div class="center-button">', unsafe_allow_html=True)
            #     if st.button("Start Speaking", key="speak_button"):
                   
            #         greeting = get_greeting()
            #         st.session_state.chat_history.append(("Chef Mate", greeting))
            #         speak(greeting)
            #         execute_voice_assistant()
            #     st.markdown('</div>', unsafe_allow_html=True)
        
        

    elif feature == "**Chat Interaction**":
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat" not in st.session_state:
            st.session_state.chat = get_conversational_chain()
        if "current_chat" not in st.session_state:
            st.session_state.current_chat = []
        if st.sidebar.button('New Chat'):
            st.session_state.current_chat = []

        for message in st.session_state.current_chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Hi, ask me anything related to cooking!")
        if user_question:
            st.chat_message("user").markdown(user_question)
            st.session_state.current_chat.append({"role": "user", "content": user_question})

            response = '\n'.join(user_input(user_question))
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.current_chat.append({"role": "assistant", "content": response})

            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.messages.append({"role": "assistant", "content": response})

            display_recipe_tip()


               

    elif feature == "**Frequently Asked Questions**":
            display_faq()
            display_recipe_tip()
       

    elif feature == "**Dish Identification**":
        st.header("🍳 Elevate your cooking experience with **Chef Mate**!👨‍🍳 Effortlessly identify dishes and ingredients, and let our expert guidance turn every meal into a culinary masterpiece. Discover the magic of effortless cooking assistance today!✨")
        input=st.text_input("Input Prompt: ",key="input")

        uploaded_file=st.file_uploader("Choose image",type=["jpg","jpeg","png"])
        image=""
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image,caption="Uploaded Image",use_column_width=True)

        submit=st.button("Find Dish")

        #if submit is clicked
        if submit:
            response=get_gemini_response(input,image)
            st.subheader("result is:")
            st.write(response)



    elif feature == "**Nutritional Analyst**":
        st.header("🍏 Unlock the secrets to healthier eating with **Chef Mate**!👨‍🍳 Analyze your ingredients and dishes for detailed nutritional insights, track calorie intake, and make every bite count towards a balanced diet. Your journey to mindful eating starts here!")
        input = st.text_input("Share additional information if you have", key="input")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        image = ""   
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)   
        submit = st.button("Find Total Calories")

        input_prompt = """
            You are a highly skilled nutritionist. You have an image of a meal, and you need to analyze the food items present in the image. For each item, provide the following information:
            1. Name of the food item
            2. Estimated calories
            3. Other nutritional details (if possible)

            Finally, calculate the total calories of the entire meal and present it in the following format:

            Itemized Nutritional Breakdown:
            1. [Food Item 1] - [Calories] calories
            2. [Food Item 2] - [Calories] calories
            ...
            n. [Food Item n] - [Calories] calories

            Total Calories: [Total Calories] calories

            Example:
            Itemized Nutritional Breakdown:
            1. Apple - 95 calories
            2. Chicken Breast - 165 calories
            3. Broccoli - 55 calories

            Total Calories: 315 calories

            Make sure to provide accurate and detailed information.
            """

        if submit:
            if uploaded_file is not None:
                try:
                    image_data = input_image_setup(uploaded_file)
                    response = gemini_repsonse(input, image_data, input_prompt)
                    # Add CSS for better styling
                    st.markdown("""
                        <style>
                            .response {
                                font-family: Arial, sans-serif;
                                background-color: #f9f9f9;
                                padding: 20px;
                                border-radius: 10px;
                                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                            }
                            .response h4 {
                                color: #007bff;
                            }
                            .response p {
                                color: #333;
                            }
                        </style>
                    """, unsafe_allow_html=True)

                    # Format the response
                    formatted_response = f"""
                    <div class="response">
                        <h4>Total Calories Calculation</h4>
                        <p>{response}</p>
                    </div>
                    """
                    st.markdown(formatted_response, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error processing the image: {e}")
            else:
                st.error("Please share image, then i can help you.")
            

    elif feature == "**Recipe Recommendations**":
        st.header("Simply type some ingredients you have on hand and **CHEFMATE** will instantly generate different recipes on demand...")

        # Collect user inputs
        st.header("1. Meal")
        meal = st.selectbox("Kind of meal", ["Select an option", "Breakfast", "Lunch", "Dinner", "Dessert", "Snack", "Drinks"])

        st.header("2. Maximum Preparation Time")
        time = st.time_input("Time you are willing to spend", datetime.time(0, 0))

        st.header("3. Ingredients to Include")
        included_ingredients = st.text_input("Include those ingredients", value="")

        st.header("4. Ingredients to Exclude")
        excluded_ingredients = st.text_input("Exclude those ingredients", value="")

        st.header("5. Describe what kind of food you are into")
        description = st.text_input("Complement the answers above")

        if st.button("Generate recommendation"):
            try:
                prompt = build_prompt(meal, time, included_ingredients, excluded_ingredients, description)
                response = llm(prompt)
                
                # Extract the content from the response object
                response_content = response.content  # or response.text or another attribute

                # Ensure response is not empty
                if not response_content.strip():
                    st.write("No recipes found. Please adjust your preferences.")
                else:
                    st.write("---")
                    st.write("### 🥘 **Recommended Recipes**")

                    # Split and display the recipes
                    recipes = response_content.split('\n\n')
                    for i, recipe in enumerate(recipes):
                        if recipe.strip():
                            st.write(f"**Recipe {i+1}:**")
                            st.write(recipe)
            except AttributeError as e:
                st.write(f"AttributeError: {e}")
            except Exception as e:
                st.write(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
