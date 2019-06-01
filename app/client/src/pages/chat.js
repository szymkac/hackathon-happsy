import React, { Component } from 'react';
import io from 'socket.io-client'
import Webcam from "react-webcam";
import { withRouter } from 'react-router-dom';
import { MainWrapper, ChatHeader, MessagesWrapper, TextWrapper, MessageInput, MessageButton, WebcamWrapper } from '../components/styled';
import Message from '../components/message'
import logo from '../emoji/logoHappsy.png';


class ChatPage extends Component {
    constructor(props) {
        super(props);
        this.messageRef = React.createRef();
        this.webcamRef = React.createRef();
        this.scrollRef = React.createRef();
    }

    state = {
        messages: []
    }

    componentDidMount() {
        document.addEventListener('keydown', this.keydownHandler);
        const nick = this.props.location && this.props.location.state ? this.props.location.state.nick : null;
        if (!!nick) {
            this.socket = io.connect('http://127.0.0.1:5000');
            this.socket.on('my response', msg => {
                console.log(msg)
                this.setState(prevState => ({ messages: prevState.messages.concat([msg]) }));
            });
        }
        else {
            this.props.history.push("/");
        }
    }

    componentWillUnmount() {
        document.removeEventListener('keydown', this.keydownHandler);
    }

    componentDidUpdate() {
        this.scrollRef.current.scrollIntoView({ behavior: 'smooth' });
    }

    keydownHandler = e => {
        if (e.keyCode === 13 && e.ctrlKey) {
            this.messageRef.current.value += '\n';
        }
        else if (e.keyCode === 13 && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    capture = () => {
        const imageSrc = this.webcamRef.current.getScreenshot();
        return imageSrc;
    };

    sendMessage = () => {
        const message = this.messageRef.current.value;
        if (!!message) {
            this.messageRef.current.value = '';
            const photo = this.capture();
            this.socket.emit('sendMessageEvent', {
                nick: this.props.location.state.nick,
                message: message,
                photo: photo
            });
        }
    }

    render() {
        const { messages } = this.state;
        const nick = this.props.location && this.props.location.state ? this.props.location.state.nick : null;

        const videoConstraints = {
            width: 1280,
            height: 720,
            facingMode: "user"
        };

        return (
            <MainWrapper>
                <ChatHeader>
                    <img src={logo} alt={"logo"} width="75px" height="75px" />
                    <h3>{nick}</h3>
                </ChatHeader>
                <MessagesWrapper>
                    {messages.map((x, i) => <Message message={x} key={i} yourNick={nick === x.nick} />)}
                    <div ref={this.scrollRef}></div>
                </MessagesWrapper>
                <TextWrapper>
                    <MessageInput ref={this.messageRef}></MessageInput>
                    <MessageButton onClick={this.sendMessage}></MessageButton>
                    <WebcamWrapper>
                        <Webcam
                            audio={false}
                            height={90}
                            ref={this.webcamRef}
                            screenshotFormat="image/jpeg"
                            width={160}
                            videoConstraints={videoConstraints}
                        />
                    </WebcamWrapper>
                </TextWrapper>
            </MainWrapper>
        );
    }
}

export default withRouter(ChatPage);