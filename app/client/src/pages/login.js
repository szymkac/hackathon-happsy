import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { MainWrapper, LoginWrapper, LoginPageLogoWrapper, LoginInput, Error } from '../components/styled'
import logo from '../emoji/logoHappsy.png';

class LoginPage extends Component {
    constructor(props) {
        super(props);
        this.nickRef = React.createRef();
    }

    state = {
        valid: true
    }

    onSubmit = (e) => {
        const nick = this.nickRef.current.value;
        if (nick !== '') {
            this.props.history.push({
                pathname: '/chat',
                state: { nick }
            });
        }
        else
            this.setState({ valid: false });

        e.preventDefault();
    }

    render() {
        const { valid } = this.state;
        return (
            <MainWrapper column>
                <LoginPageLogoWrapper>
                    <img src={logo} alt={"logo"} width="100px" height="100px" />
                </LoginPageLogoWrapper>
                <LoginWrapper>
                    <h1>Login</h1>
                    <form onSubmit={this.onSubmit}>
                        <label>Temporary Nick:
                            <br />
                            <LoginInput first type="text" ref={this.nickRef} />
                        </label>
                        <LoginInput type="submit" value="Enter Happsy" />
                        {valid || <Error>Nick is required!!</Error>}
                    </form>
                </LoginWrapper>
            </MainWrapper>
        );
    }
}

export default withRouter(LoginPage);