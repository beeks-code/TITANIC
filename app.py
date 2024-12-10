import streamlit as st
import pandas as pd
import pickle
import numpy as np
if_survived = np.array([
    "🎉 You survived! Guess you're a natural at dodging icebergs. 🛳️",
    "🥳 Well done! You’re officially unsinkable. 🚢",
    "😅 Phew! You escaped the iceberg and the cold water. Nice work! ❄️",
    "💃 You made it! Now, go find Jack’s sketchpad. 🎨",
    "🕴️ Survived? Well, clearly you're the James Bond of the Titanic. 🕶️",
    "🍸 Congrats! You survived the Titanic – now survive the afterparty! 🥂",
    "🥇 Survivor’s Club member: Level Expert. 🏆",
    "🛶 Who needs a lifeboat when you've got survival skills like yours? 👑",
    "🍀 You’ve got the luck of the sea. The iceberg never stood a chance! 🌊",
    "❄️ Iceberg, meet the legend. You’re officially unsinkable! 🏄‍♂️",
    "🍹 Survived! Now, go grab a cocktail – you earned it. 🍸",
    "🦸‍♂️ You dodged the iceberg. You’re basically a Titanic superhero! 🦸‍♀️",
    "🌊 You made it! Now, go tell the iceberg to take a hike. 🏞️",
    "🎩 Titanic who? You’re too cool to sink! 😎",
    "🎮 Survival skills unlocked! Just avoid the next iceberg, okay? 🧊",
    "🏅 You outsmarted the iceberg. You deserve a medal! 🏅",
    "🎉 Congrats! You’ve got the best lifeboat instincts in the world. 🛥️",
    "🌴 Survived the Titanic – what’s next? Maybe a cruise to the Bahamas? 🏖️",
    "🧊 The iceberg is probably still asking, 'How did they do that?' 🤔",
    "👏 Well, look at you – escaping icebergs like a pro! 🌨️",
    "🎓 You survived! Looks like you're a Titanic expert in the making. 📚",
    "🚴‍♂️ You’ve got the moves of a pro iceberg dodger! 🏃‍♂️",
    "🏆 Congratulations! You’re now officially a Titanic legend! 🌟",
    "🏅 You survived, but can you survive this quiz next? 🤓",
    "🍿 You outran the iceberg! Time to celebrate with some shipwrecked snacks. 🍪",
    "🛥️ Who needs a lifeboat when you’ve got skills like that? 👏",
    "🍾 You survived! If only the iceberg had your survival instincts. 🎯",
    "🧊 Congrats! You’ve got the survival skills of a pro icebreaker! 🛳️",
    "🦸‍♀️ You made it! That iceberg didn’t know who it was messing with. 🔥",
    "🥇 Survived? More like ‘Titanic Champion.’ Now where’s your victory lap? 🏁",
])

if_not_survived = np.array([
    "☠️ The Titanic claimed another soul. Better luck in your next life. ⛴️",
    "⚰️ Looks like the iceberg got you first. Sorry about that. 🧊",
    "💀 Titanic’s ‘unsinkable’ reputation was wrong about you. ⛴️",
    "🕯️ It's a wrap! You were the first to go, RIP. 🧟‍♀️",
    "😱 Looks like the iceberg had a more permanent reservation for you. 🏝️",
    "🎩 You should have boarded the lifeboat! Should've known better. ⛴️",
    "👻 Ghosted by the iceberg. That's tough. 💀",
    "💔 The Titanic took you down, but don't worry, you're now a legend. 🌊",
    "🦴 Looks like you didn't make the cut for the lifeboat club. 🚤",
    "💣 The Titanic's list of survivors didn’t include you. Plot twist: You're history. 📜",
    "🕵️‍♂️ You could've been the plot twist, but instead, you were the first casualty. 💥",
    "⚰️ It’s not just the iceberg, your fate was sealed the moment you boarded. 🎲",
    "🎭 Your death scene was dramatic, but you never made it to the sequel. 🎬",
    "🌑 You went down with the ship, but at least you're not alone... in the afterlife. 💀",
    "🎲 Unfortunately, your number was up when the ship went down. 🛳️",
    "🏴‍☠️ The Titanic sank, but your hopes went down faster. ⛵",
    "🥀 The iceberg wasn’t the only thing that was cold that night. 🧊",
    "⚔️ You tried, but the iceberg was just too savage. 🦸‍♂️",
    "🧛‍♂️ You’ve become a vampire of the sea, forever wandering the Titanic's ghost ship. 🌊",
    "💀 You didn’t survive, but at least you can haunt the ship. 👻",
    "🌪️ You didn’t make it, but the iceberg surely did. 🌬️",
    "😈 The Titanic was a one-way trip to the afterlife for you. 🛑",
    "🛑 The Titanic said no. Your survival rate was zero. 📉",
    "💨 The iceberg’s chill got you, but you never had a chance to warm up. 🥶",
    "🎬 The Titanic may have sank, but your death was Oscar-worthy. 🎥",
    "🔮 Fate wasn’t on your side. The iceberg knew. 🧙‍♀️",
    "💀 Not everyone is meant to make it. You were the first to find that out. 👀",
    "🕰️ It wasn’t the clock that sank the ship, it was your bad luck. ⏳",
    "⚰️ Here lies a Titanic failure. 🏴‍☠️ RIP ⚰️",
    "🦦 The iceberg decided you were the appetizer, not the main course. 🍽️",
    "🎩 You’re part of Titanic's tragic history now. A story for the ages. 📖",
    "💀 One minute you’re on top of the world, the next you’re at the bottom of the ocean. 🌊",
])

st.set_page_config(page_title="Titanic Test Webapp", page_icon="🚢", layout="centered")

# Welcome message
st.title("🚢 Titanic Prediction Webapp")
st.write("Hey, Welcome! ")
survive_msg=np.random.choice(if_survived)
death_msg=np.random.choice(if_not_survived)
# Set page configuration


if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Button-like navigation
st.markdown("""
    <style>
        .stButton>button {
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)
if st.session_state.get('current_page') == "Home":
    st.write("Choose an option:")

    if st.button("📊 Use Model"):
        st.session_state.current_page = "Use Model"
    if st.button("📈 Preview On Titanic Dataset"):
        st.session_state.current_page = "Preview On Titanic Dataset"
    if st.button("ℹ️ About This Works"):
        st.session_state.current_page = "About This Works"

elif st.session_state.get('current_page') == "About This Works":
    st.subheader("About This Works")
    st.write("""
       Welcome to the Titanic Survival Prediction app! 🚢
        
        This web application predicts the likelihood of survival for a Titanic passenger based on several key factors, such as:
        - **Passenger Class (Pclass)**: Whether the passenger was traveling first, second, or third class.
        - **Age**: The passenger's age at the time of the voyage.
        - **Sex**: Whether the passenger was male or female.
        - **Fare**: The amount the passenger paid for their ticket.
        - **Port of Embarkation**: The location from which the passenger boarded the Titanic (Cherbourg, Queenstown, or Southampton).

        The prediction is made using a **Decision Tree classifier**, a popular machine learning algorithm trained on historical data from the Titanic disaster. By analyzing this data, the model has learned to predict whether a passenger would have survived the tragic sinking based on these attributes.

        **How does the prediction work?**
        - Once you enter your details, the model processes them and generates a prediction, giving you a likelihood of survival based on historical patterns. The more accurate the inputs, the more reliable the prediction.
        
        **Note**: While the model is trained on real historical data, this is for entertainment purposes, and the results are purely based on the factors listed. So, if you survive, you might just be channeling your inner Jack Dawson. If not, well, you can always try again! 😅
    """)
    st.write("**By:Beekrm**")

    # Home button
    if st.button("🏠 Go Back Home"):
        st.session_state.current_page = "Home"
elif st.session_state.get('current_page') == "Use Model":
    # Load model and setup prediction form (current prediction code)
    
    # Load the trained model pipeline
    with open('titanic_pipeline.pkl', 'rb') as file:

        model_pipeline = pickle.load(file)

    st.write("Enter your details below to predict whether you would have survived:")

    # User Input Section for Prediction
    name = st.text_input("Your Name", "")
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"], index=0)
    age = st.number_input("Age (in years)", min_value=1, max_value=100, value=22)
    fare = st.number_input("Fare (in USD)", min_value=0.0, value=7.25, step=0.1)
    embarked = st.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"], index=2)

    # Map input values to match dataset columns
    embarked_map = {"C (Cherbourg)": "C", "Q (Queenstown)": "Q", "S (Southampton)": "S"}
    embarked = embarked_map[embarked]

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Predict button
    if st.button("Predict Survival"):
        # Make prediction using the trained model
        prediction = model_pipeline.predict(input_data)
        result = "You Would Have Survived 🟢" if prediction[0] == 1 else "Wouldn't Have Survived 🔴"

        # Display result with styling
        if prediction[0] == 1:
            st.markdown(f"<h2 style='text-align: center; color: green;'>{result}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: grey;'> {survive_msg} </h3>", unsafe_allow_html=True)
            st.balloons()  # Survival animation (balloons)
        else:
            st.markdown(f"<h2 style='text-align: center; color: red;'>{result}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: grey;'> {death_msg} </h3>", unsafe_allow_html=True)
            st.write(f"**Rest in Peace  {name}.**")
            # st.image("titanic_project/image.jpg",width=400)


    # Home button
    if st.button("🏠 Go Back Home"):
        st.session_state.current_page = "Home"

elif st.session_state.get('current_page') == "Preview On Titanic Dataset":
    # Load the Titanic dataset (replace with the actual path to your Titanic dataset)
    df = pd.read_csv(r"https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    
    # Provide options for the user to choose
    preview_option = st.selectbox(
        "Choose a preview option:",
        ["Head of the dataset", "Tail of the dataset", "Random sample of the dataset"]
    )
    
    if preview_option == "Head of the dataset":
        # Slider to select the number of rows for the head preview
        num_head_rows = st.slider("Select the number of rows from the head:", 1, len(df), 5)
        st.write(f"Here’s the first {num_head_rows} rows of the Titanic dataset:")
        st.dataframe(df.head(num_head_rows))  # Display selected number of rows from the head
    
    elif preview_option == "Tail of the dataset":
        # Slider to select the number of rows for the tail preview
        num_tail_rows = st.slider("Select the number of rows from the tail:", 1, len(df), 5)
        st.write(f"Here’s the last {num_tail_rows} rows of the Titanic dataset:")
        st.dataframe(df.tail(num_tail_rows))  # Display selected number of rows from the tail
    
    elif preview_option == "Random sample of the dataset":
        # Ask for the number of random rows to display
        num_samples = st.slider("Select the number of random samples:", 1, 100, 5)
        
        # Button to regenerate random sample
        if st.button("Generate Random Sample"):
            st.write(f"Here’s a new random sample of {num_samples} rows from the Titanic dataset:")
            st.write("Dataset Overview:")
            st.dataframe(df.sample(num_samples))  # Display a new random sample of rows
    
    # Home button
    if st.button("🏠 Go Back Home"):
        st.session_state.current_page = "Home"