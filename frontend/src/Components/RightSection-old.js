import React, { useState } from "react";
import { createSearchParams, Link, useNavigate } from "react-router-dom";
import { API_URL } from "../utils/constants";
import female from '../images/female.png';
import male from '../images/male.png';
import APIService from '../Api/APIService'


/**
 * initial states for inputs
 */
const initialState = { 
        Age: "",
        Sex: "",
        ChestPainType:"",
        RestingBP:"",
        Cholesterol:"",
        FastingBS:"",
        RestingECG:"",
        MaxHR:"",
        Oldpeak:"",
        ExerciseAngina:"",
        ST_Slope:""
    };


const  RightSection = () => {

    //useStates
     const [formData, setFormData] = useState(initialState); //form date 
     const navigate = useNavigate(); //navigate to other pages
    /**
     * input onChange method
     * @param {e} e 
     */
    const onChange = (e) => {
        const { name, value } = e.target;  
        setFormData({ ...formData, [e.target.name]: e.target.value });
        console.log(e.target.name + " : " + e.target.value)
    };

    /**
     * submit form data to backend
     */
    const submit =async (e) =>{
        if(formData.ChestPainType === ""){
          formData.ChestPainType = "1";
        }
        if(formData.ST_Slope === ""){
          formData.ST_Slope = "1"
        }
        if(formData.RestingECG === ""){
          formData.RestingECG = "1"
        }
        if(formData){
             try {
            const rawResponse = await fetch(`http://127.0.0.1:5000/heart?Age=`+formData.Age+`&Sex=`+formData.Sex+`&ChestPainType=`+formData.ChestPainType+`&RestingBP=`+formData.RestingBP+`&Cholesterol=`+formData.Cholesterol+`&FastingBS=`+formData.FastingBS+`&RestingECG=`+formData.RestingECG+`&MaxHR=`+formData.MaxHR+`&ExerciseAngina=`+formData.ExerciseAngina+`&Oldpeak=`+formData.Oldpeak+`&ST_Slope=`+formData.ST_Slope+``, {
              method: 'POST',
              mode: 'cors',
              headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
              },
          })
              const content = await rawResponse.json();
                   
                 navigate({
                    pathname: "/finish",
                    search:createSearchParams({
                      result: content.HeartDisease
                    }).toString()
                });
        
             } catch (error) {
                error.response && console.log(error.response.data);
             }
        }
    }

  return (
  <div className="form_container">
        <div className="title">
          Heart disease prediction
        </div>
        <form action="">
          <div className="form_fields">

            <div className="input_group">
              <div className="input_container">
                <label for="Age">What is your Age </label>
                <input list="ages"  type="number" id="Age" name="Age" placeholder=" Enter your Age" min={1} max="100" required onChange={onChange}/>
                
              </div>
              <div className="input_container">
                <label for="Sex">Select your Sex</label>
                <div className="radio_group">
                    <div className="label_wrap">
                         <label className={formData.Sex == "1"? "radioMaleLabelChecked" : "radioMaleLabel"} for="genderMale"><img src={male} width="30"/></label>
                        <input className="radio" type="radio" id="genderMale" name="Sex" value="1" required onChange={onChange}/>
                    </div>
                    
                    <div className="label_wrap">
                         <label className={formData.Sex == "2"? "radioFemaleLabelChecked" : "radioFemaleLabel"} for="genderFemale"><img src={female} width="30"/></label>
                        <input type="radio" className="radio" id="genderFemale" name="Sex" value="2" required onChange={onChange}/>
                    </div>
                </div>
              </div>
            </div>
 
              <div className="input_container">
                <label for="ChestPainType">What is your chest pain type </label>
                 <select name="ChestPainType" id="ChestPainType" onChange={onChange}>
                    <option value="3">ATA</option>
                    <option value="2">NAP</option>
                    <option value="1">ASY</option>
                    <option value="4">TA</option>
                </select>
              </div>
              <div className="input_container">
                <label for="RestingBP">What is your resting BP count </label>
                <input type="number" id="RestingBP" name="RestingBP" placeholder=" Enter resting BP count"  required onChange={onChange}/>
              </div> 

            <div className="input_group">
              <div className="input_container">
                <label for="Cholesterol">What is your Cholesterol level </label>
                <input type="number" id="Cholesterol" name="Cholesterol" placeholder=" Enter your Cholesterol level " required onChange={onChange}/>
              </div>
              <div className="input_container">
                <label for="FastingBS">What is your fasting BS  </label>
                <input type="number" id="FastingBS" name="FastingBS" placeholder=" Enter fasting BS" min={0} max={1} required onChange={onChange}/>
              </div>
            </div>

            <div className="input_group">
              <div className="input_container">
                <label for="RestingECG">What is your resting ECG state </label>
                <select name="RestingECG" id="RestingECG" onChange={onChange}>
                    <option value="3">Normal</option>
                    <option value="2">ST</option>
                    <option value="1">LVH</option>
                </select>
              </div>
              <div className="input_container">
                <label for="MaxHR">What is your max HR </label>
                <input type="number" id="MaxHR" name="MaxHR" placeholder=" Enter max HR"  required onChange={onChange}/>
              </div>
            </div>

             <div className="input_container">
                <label for="Oldpeak">What is your old peak count  </label>
                <input type="number" id="Oldpeak" name="Oldpeak" placeholder=" Enter old peak"  required onChange={onChange}/>
              </div>
              <div className="input_container">
                <label for="ExerciseAngina">Do you have exercise angina ? </label>
                <div className="radio_group">
                    <div className="label_wrap">
                        <label for="exerciseAnginaYes">Yes </label>
                        <input type="radio" id="exerciseAnginaYes" name="ExerciseAngina" value="2" required onChange={onChange}/>
                    </div>
                    <div className="label_wrap">
                        <label for="exerciseAnginaNo">No </label>
                        <input type="radio" id="exerciseAnginaNo" name="ExerciseAngina" value="1" required onChange={onChange}/>
                    </div>
                </div>
              </div>

            
              <div className="input_container">
                <label for="ST_Slope">What is your ST Slope type </label>
                <select name="ST_Slope" id="ST_Slope" onChange={onChange}>
                    <option value="3">Up</option>
                    <option value="2">Flat</option>
                    <option value="1">Down</option>
                </select>
              </div>

            <div className="input_container submit">
              <input type="button" value="Predict with Einstein" className="submit-btn" onClick={(e)=>submit(e)}/>
            </div>

            <div className="input_container submit">
              <input type="submit" value="Predict with Terra predictor" className="submit-btn" onClick={(e)=>submit(e)}/>
            </div>

          </div>
        </form>
      </div>
  );
}

export default RightSection;