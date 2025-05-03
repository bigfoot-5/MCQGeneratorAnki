# MCQ Generator with AI for Vocabulary Practice

An Anki add-on to automatically generate multiple-choice vocabulary questions using AI.


## ✨ Features

* AI-generated sentence with blank using your word
* Local distractors from other deck words
* One-click generation for selected cards or entire deck
* Works with AnkiDroid after syncing
* Customizable via `config.json`


## 🛠 Installation

### 1. Install the Note Type

Go to **File → Import**, and import the provided `MCQ Basic.apkg`. You will get a new note type named **MCQ Basic**. Select this note type while making new notes and fill out only the **Word** field with your desired word or phrase, and leave other fields blank.


### 2. Install the Add-on

* **Via AnkiWeb**: Use the code (TBD) in **Tools → Add-ons → Get Add-ons…**
* **Manual**: Clone this repo into your `addons21` folder or use "Install from file…" with the `mcq_generator.ankiaddon`.


## ⚙️ Configuration

Edit `config.json`:

```json
{
  "api_key": "your_api_key_here",
  "api_url": "https://api.groq.com/openai/v1/chat/completions",
  "model": "your_chosen_model",
  "prompt_template": "Generate a normal length English sentence using the word or phrase '{word}', replacing the target word or phrase with a blank (_____). Difficulty should be {level} based on CEFR. Return only the sentence.",
  "temperature": 1.5
}
```


## ▶️ How to Use

1. Fill **Word** field in MCQ Basic notes.
2. For generating specific notes of a deck:
    1. Open the **Browser**
    2. Select the notes you want to make MCQs out of.
    3. Then, **Edit → Generate MCQ (selected notes)** — for selected cards.
3. For generating MCQs for a whole deck: **Tools → Generate MCQ (whole deck) → Select your deck → Press OK**.


## 📱 Mobile

Just sync your collection to AnkiWeb to review from mobile clients, e.g. AnkiDroid or AnkiMobile. Cards appear as interactive MCQs.


## 📎 License & Contributions

This project is licensed under the Apache License 2.0.