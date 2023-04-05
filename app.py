import pickle
import gradio as gr
import numpy as np
import sklearn

# decleration of the params
model_path = "1_IrısClassification/loj_reg.sav"#'loj_reg.sav'
model = pickle.load(open(model_path, 'rb'))
classes = {
    0:'setosa',
    1:'versicolor',
    2:'virginica',
}

examples = [
    [5.1,	3.4, 1.5,	0.2],
    [6.1, 3. , 4.6, 1.4],
    [6.8, 3. , 5.5, 2.1],
    [4.4, 2.9, 1.4, 0.2],
    [5.8, 2.7, 3.9, 1.2],
    [6.1, 3. , 4.9, 1.8]
]

# Function which uses the model
def predict(sepal_length, sepal_width, petal_length, petal_width):
    # preparing the input into convenient form
    features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(-1,4)
    
    # prediction
    probs = model.predict_proba(features)[0]
    
    # adjusting the output
    class_probabilities = dict(zip(classes.values(), np.round(probs,3)))
    
    return class_probabilities

# input components
sepal_length = gr.inputs.Slider(minimum=0.1, maximum=10, default=5.1, label = 'sepal_length')
sepal_width = gr.inputs.Slider(minimum=1, maximum=10, default=3.5, label = 'sepal_width')
petal_length = gr.inputs.Slider(minimum=1, maximum=10, default=1.4	, label = 'petal_length')
petal_width = gr.inputs.Slider(minimum=1, maximum=5, default=0.2, label = 'petal_width')

# declerating the params
demo_params = {
    "fn":predict,
    "inputs":[sepal_length,sepal_width,petal_length,petal_width], 
    "outputs":"label",
    "examples":examples,
    "cache_examples":True
}

# Creating application
demo_app = gr.Interface(**demo_params)

if __name__ == "__main__":
    demo_app.launch()