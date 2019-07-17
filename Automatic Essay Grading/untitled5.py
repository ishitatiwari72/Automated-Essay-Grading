# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 03:25:45 2019

@author: lenovo
"""

from flask import Flask, request, render_template, url_for
from flask_work import Flask_srk

app = Flask(__name__)

@app.route("/main")
def home():
    return render_template("index.html")

@app.route("/result",methods=["POST"])
def output():
    form_data = request.form
    status = Flask_srk(form_data["antshant"])
    print(status)
    return render_template("response.html",status1=str(status[0]),status2=str(status[1]),status3=str(status[2]),status4=str(status[3]),status5=str(status[4]),status6=str(status[5]),status7=str(status[6]),status8=str(status[7]),status9=str(status[8]),status10=str(status[9]),status11=str(status[10]))

if __name__ == "__main__":
    app.run(port=8000)
    
    

