import React from 'react';
import { SingleMessageWrapper, MessageContainer, MessageBoxItem, MessageNick, MessageParagraph } from './styled';
import happyImage from '../emoji/happy.png';
import sadImage from '../emoji/sad.png';
import angryImage from '../emoji/angry.png';
import surpriseImage from '../emoji/surprise.png';
import neutralImage from '../emoji/neutral.png';
import nothingImage from '../emoji/nothing.png';

const Message = ({ message, yourNick }) => {
    let image;

    switch (message.emotion) {
        case "happy":
            image = happyImage;
            break;
        case "sadness":
            image = sadImage;
            break;
        case "anger":
            image = angryImage;
            break;
        case "surprise":
            image = surpriseImage;
            break;
        case "neutral":
            image = neutralImage;
            break;
        default:
            image = nothingImage;
            break;
    }

    const messageText = <MessageBoxItem>
        <MessageNick>{message.nick}</MessageNick>
        {message.message.split('\n').map((m, i) => <MessageParagraph key={i}>{m}</MessageParagraph>)}
    </MessageBoxItem>;

    const messageEmoji = <MessageBoxItem emoji={true}>
        <img src={image} alt={message.emotion} width="75px" height="75px" />
    </MessageBoxItem>;

    const content = yourNick ?
        <React.Fragment>
            {messageText}
            {messageEmoji}
        </React.Fragment> :
        <React.Fragment>
            {messageEmoji}
            {messageText}
        </React.Fragment>;

    return (
        <SingleMessageWrapper yourNick={yourNick}>
            <MessageContainer yourNick={yourNick}>
                {content}
            </MessageContainer>
        </SingleMessageWrapper>
    );
}

export default Message;