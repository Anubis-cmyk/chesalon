import React, { useState } from "react";
import { useNavigate,useSearchParams } from "react-router-dom";
import resultGood from "../images/resultgood.png";
import resultBad from "../images/resultbad.png";

const  FinishPage = () => {
    const navigate = useNavigate(); //navigate to other pages
    const [searchParams] = useSearchParams();
    const [result, setResult] = useState(searchParams.get("result"));
    
    return(
        <div className="resultBack">
            <div className={result*100 > 50 ? "form_container result-container resultBackBad" : "form_container result-container resultBackGood"}>
                <img src={result*100 > 50 ?resultBad : resultGood} width="50%"/>
                <div className="title">
                    <h2>Predicted result is</h2>
                    <div className="para">
                        <p className={result*100 > 50 ?"text-red":"text-green"}><h1>We predict you have {result*100}% chance to have heart disease </h1> </p>
                    </div>
                    <h4>Thank you </h4>
                </div>
                <button className={result*100 > 50 ? "backButton btnRed" : "backButton btnGreen"} onClick={()=> navigate("/")}>Back</button>
            </div>
        </div>
    )
}

export default FinishPage;