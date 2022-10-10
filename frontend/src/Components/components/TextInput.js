import React from "react";

const TextInput = (type,name,id,placeholder,required) =>{
    return(
        <div>
            <input type={type} name={name} placeholder={placeholder} required={required} id={id} />
        </div>
    )
}

export default TextInput;