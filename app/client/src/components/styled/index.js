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

    ${props => props.column && css`
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    `
    }
`;

const MiddleWrapper = styled.div`
    height:100%;
`;

const ChatHeader = styled.div`
    height: 150px;
    display: flex;
    justify-content: center;
    background: #5b9bd5;
    color: #ffffff;
`;

const HeaderLogoWrapper = styled.div`
    align-self: center;
    margin-left: calc(50% - 50px);
`;

const HeaderUserInfoWrapper = styled.div`
    align-self: center;
    margin-left: auto;
    text-align: left;
`;

const MessagesWrapper = styled.div`
    height: calc(100% - 260px);
    overflow-y: scroll;
    background: #ffffff;
    border-top-left-radius: 20px;
    border: solid 4px #ffce24;
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
    border-bottom-left-radius: 20px;
    border: solid 4px #ffce24;
    resize: none;
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
    border: solid #dddddd 2px;
    border-radius: 20px;
    background: lightgray;
    display: flex;
    ${props => props.yourNick ?
        css`
            border-bottom-left-radius: 0;
            padding: 20px 10px 20px 20px;
            background: #e5e5e5;
        `:
        css`
            border-bottom-right-radius: 0;
            padding: 20px 20px 20px 10px;
            background: #dbedff;
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

const FancyButton = styled.button`
    color: #ffffff;
    background: transparent;
    border: solid #ffffff 2px;
    border-radius: 20px;
    padding: 6px;
    font-weight: bold;
`;

const LoginWrapper = styled.div`
    background: #ffffff;
    border-radius: 20px;
    border: solid 4px #ffce24;
    box-sizing: border-box;
    padding: 30px;
    padding-top: 35px;
    position: relative;
    top: -45px;
    height: 255px;
`;

const LoginPageLogoWrapper = styled.div`
    overflow: visible;
    z-index: 10;
`;

const LoginInput = styled.input`
    margin: 10px 0;
    border: solid 2px #ffce24;
    background: #ffffff;
    padding: 6px;
    font-weight: bold;
    ${props => props.first && 'margin-left: 20px;'}
`;

const Error = styled.p`
    color: #c80a0a;
    font-weight: bold;
    margin: 2px 0 0 20px;
`;

export {
    BodyWrapper,
    MainWrapper,
    MiddleWrapper,
    ChatHeader,
    HeaderLogoWrapper,
    HeaderUserInfoWrapper,
    MessagesWrapper,
    TextWrapper,
    MessageInput,
    SingleMessageWrapper,
    MessageContainer,
    MessageBoxItem,
    WebcamWrapper,
    MessageNick,
    MessageParagraph,
    FancyButton,
    LoginWrapper,
    LoginPageLogoWrapper,
    LoginInput,
    Error
};