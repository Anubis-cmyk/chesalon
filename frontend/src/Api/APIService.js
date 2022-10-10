export default class APIService{
    // Insert an article
    static InsertArticle(body){
        return fetch(`{http://192.168.1.102:5000/heart?Age =40&Sex=1&ChestPainType=3&RestingBP=140&Cholesterol=289&FastingBS=0&RestingECG=3&MaxHR=172&ExerciseAngina=1&Oldpeak=0.0&ST_Slope=3`,{
            'method':'POST'
    })
    .then(response => response.json())
    .catch(error => console.log(error))
    }
}
