from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Use the Ellbendls/Qwen-2.5-3b-Text_to_SQL model
model_name = "Ellbendls/Qwen-2.5-3b-Text_to_SQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 512)

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate SQL query
    outputs = model.generate(**inputs, max_length=max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Prepare JSON response
    json_response = {"text": response}
    print(json_response)  # Print the JSON response to the console

    return jsonify(json_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)