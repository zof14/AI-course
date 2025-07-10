from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

convers_history = []
while True:
    #history of conversation

    history ="\n".join(convers_history)

    #get user input
    input_user= input("> ")

    #gokenize input
    inputs = tokenizer.encode_plus(history, input_user, return_tensors="pt")
    
    #generate response
    outputs = model.generate(**inputs)
    
    #decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    #save to history
    convers_history.append(input_user)
    convers_history.append(response)