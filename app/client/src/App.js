import React,{Component} from 'react';
import logo from './logo.svg';
import './App.css';
import Webcam from "react-webcam";

class App extends Component {
	setRef = webcam => {
    this.webcam = webcam;
  };
  
  capture = () => {
    const imageSrc = this.webcam.getScreenshot();
	console.log(imageSrc);
  };
	
  render() {
	const videoConstraints = {
      width: 1280,
      height: 720,
      facingMode: "user"
    };
	  
	  return (
		<div className="App">
			<Webcam
			  audio={false}
			  height={350}
			  ref={this.setRef}
			  screenshotFormat="image/jpeg"
			  width={350}
			  videoConstraints={videoConstraints}
			/>
			<button onClick={this.capture}>Capture photo</button>
		</div>
	  ); 
  }
}



export default App;
