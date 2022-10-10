import React, { useEffect, useState } from "react";
import { createSearchParams, Link, useNavigate } from "react-router-dom";
import { API_URL } from "../utils/constants";
import female from '../images/female.png';
import male from '../images/male.png';
import APIService from '../Api/APIService'
import TextInput from "./components/TextInput";


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
     const [inputs, setInputs] = useState([]); //all form inputs 
     const [numericInputs, setNumericInputs] = useState([])//numeric form inputs 
     const [data, setData] = useState({})//non numeric form inputs 
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

     const fetchData = async ()=>{
        const cols = await fetch(`http://127.0.0.1:5000/getData`);
        const content = await cols.json();
        setInputs(content.columns);
        setNumericInputs(content.numericColumns);
        setData(content.data);
        console.log(content.data)
      }

    useEffect( () => {
     
      fetchData()
    }, []);

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
            {inputs.map((input, index) => {
              return (
                <div key={index}>
                  {
                     data[input].type === 'text'?(
                      <div className="input_container">
                        <label for={input}>What is your {input} </label>
                        <input type="text" id={input} name={input} placeholder={" Enter your " + input} required onChange={onChange}/>
                      </div>
                     ): data[input].type === 'number'?(
                      <div className="input_container">
                        <label for={input}>What is your {input} </label>
                        <input type="number" id={input} name={input} placeholder={" Enter your " + input} required onChange={onChange}/>
                      </div>
                     ): data[input].type === 'radio'?(
                      <div className="input_container">
                         <label for={input}>What is your {input} </label>
                          <div className="label_wrap">
                        {
                            data[input].data.map((value, key) =>{
                              return(
                                <>
                                  <label for={value}>{value} </label>
                                  <input type="radio" id={value} name={input} value={key} required onChange={onChange}/>
                                </>
                            )})
                           }
                           </div>
                      </div>
                     ): data[input].type === 'select'?(
                      <div className="input_container">
                        <label for={input}>What is your {input} </label>
                        <select name={data[input].name} id={data[input].name} onChange={onChange}>
                           {
                            data[input].data.map((value, key) =>{
                              return(
                                <option value={key}>{value}</option>
                              )
                            })
                           }
                        </select>
                      </div>
                     ):(
                      <p>Not a valid data type</p>
                     )
                  }
                   
                 
                </div>
              )
            })}
       
            <div className="input_container submit">
              <input type="button" value="Predict with Einstein" className="submit-btn" onClick={()=>{}}/>
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