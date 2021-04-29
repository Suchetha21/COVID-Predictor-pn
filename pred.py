from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("symptoms_model.sav")
@app.route('/')
def home():
    return render_template("page.html")

@app.route('/predict', methods=['POST'])
def result():
    name=request.form['full_name']
    Age= float(request.form['age'])
    if Age==0:
        age_dont_know=1
    else:
        age_dont_know=0
    if Age>60:
        age_yes=1
    else:
        age_yes=0
    gender= float(request.form['gender'])
    if gender==0:
        male=0
        other=0
        gender='female'
    elif gender==1:
        male=1
        other=0
        gender='male'
    else:
        male=0
        other=1
        gender='others'
    Symptoms=request.form.getlist('Symptoms')
    if 'cough' in Symptoms:
        cough=1
    else:
        cough=0
    if 'fever' in Symptoms:
        fever=1
    else:
        fever=0
    if 'sore_throat' in Symptoms:
        sore_throat=1
    else:
        sore_throat=0
    if 'shortness_of_breath' in Symptoms:
        shortness_of_breath=1
    else:
        shortness_of_breath=0
    if 'head_ache' in Symptoms:
        head_ache=1
    else:
        head_ache=0
    
    test_indicator= float(request.form['test_indicator'])
    if test_indicator==0:
        contact_with_confirmed=0
        others=0
    elif test_indicator==1:
        contact_with_confirmed=1
        others=0
    else:
        contact_with_confirmed=0
        others=1

    val=np.array([[int(cough),int(fever),int(sore_throat),int(shortness_of_breath),int(head_ache),int(age_yes),int(age_dont_know),int(male),int(other),int(contact_with_confirmed),int(others)]])

    #pred=model.predict(np.array([[int(cough),int(fever),int(sore_throat),int(shortness_of_breath),int(head_ache),int(age_yes),int(age_dont_know),int(male),int(other),int(contact_with_confirmed),int(others)]]))[0]
    pred=model.predict(val)[0]
    predprob=model.predict_proba(val)[0][2]*100
    print(predprob)


    
    return render_template("result.html", result =pred , result1=round(predprob,3),name=name,Age=int(Age) ,gender=gender)
    
    

if __name__ == '__main__':
    app.run()