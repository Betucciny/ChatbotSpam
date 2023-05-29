from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from cleaner import clean_corpus
from chatterbot.trainers import ChatterBotCorpusTrainer


chatbot = ChatBot("Chatpot")

files = ['chatbotdata/chat1.txt']
# trainer1 = ChatterBotCorpusTrainer(chatbot)
# trainer1.train("chatterbot.corpus.spanish")

trainer2 = ListTrainer(chatbot)

for file in files:
    cleaned = clean_corpus(file)
    trainer2.train(cleaned)

exit_conditions = (":q", "quit", "exit")
while True:
    query = input("> ")
    if query in exit_conditions:
        break
    else:
        print(f"🪴 {chatbot.get_response(query)}")
