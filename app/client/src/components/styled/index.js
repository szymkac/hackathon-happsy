import styled, { css } from 'styled-components'

const BodyWrapper = styled.div`
    height: 100vh;
    width: 100vw;
    overflow: auto;
`;

const MainWrapper = styled.div`
    min-width: 500px;
    max-width: 900px;
    width: 60%;
    height: 100%;
    margin: 0 auto;
`;

const MiddleWrapper = styled.div`
    height:100%;
`;

const ChatHeader = styled.div`
    height: 150px;
`;

const MessagesWrapper = styled.div`
    height: calc(100% - 260px);
    overflow-y: scroll;

`;

const TextWrapper = styled.div`
    display: flex;
`;

const MessageInput = styled.textarea`
    box-sizing: border-box;
    min-width: 290px;
    flex-grow: 1;
    flex-shrink: 1;
    height: 90px;
    margin: 10px 0;
`;

const MessageButton = styled.button`
    min-width: 50px;
    height: 90px;
    margin: 10px 0;
`;

const SingleMessageWrapper = styled.div`
    width: 100%;
    display: flex;
    justify-content: ${props => props.yourNick ? 'flex-end' : 'flex-start'};
`;

const MessageContainer = styled.div`
    width: 60%;
    margin: 5px;
    box-sizing: border-box;
    border: solid black 1px;
    border-radius: 20px;
    background: lightgray;
    display: flex;
    ${props => props.yourNick ?
        css`
            border-bottom-left-radius: 0;
            padding: 20px 10px 20px 20px;
        `:
        css`
            border-bottom-right-radius: 0;
            padding: 20px 20px 20px 10px;
        `}
`;

const MessageBoxItem = styled.div`
    display:flex;
    justify-content: center
    ${props => props.emoji ?
        css`
            flex-grow: 0;
            justify-content: center;
            align-items: center;
            min-width: 75px;
        `:
        css`
            flex-grow: 1;
            justify-content: start;
            align-items:start;
            flex-direction: column;
        `}
`;

const WebcamWrapper = styled.div`
    margin: 10px 0;
`;

const MessageNick = styled.p`
    font-weight: bold;
    margin: 0;
    word-wrap: break-word;
    max-width: 100%;
`;

const MessageParagraph = styled.p`
    margin: 1px 0;
    word-wrap: break-word;
    max-width: 100%;
`;

export {
    BodyWrapper,
    MainWrapper,
    MiddleWrapper,
    ChatHeader,
    MessagesWrapper,
    TextWrapper,
    MessageInput,
    MessageButton,
    SingleMessageWrapper,
    MessageContainer,
    MessageBoxItem,
    WebcamWrapper,
    MessageNick,
    MessageParagraph
};