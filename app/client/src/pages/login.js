import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';

class LoginPage extends Component {
    constructor(props) {
        super(props);
        this.nickRef = React.createRef();
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
            alert("fill nick!!");
        e.preventDefault();
    }
    render() {
        return (
            <div>
                <h1>Login</h1>
                <form onSubmit={this.onSubmit}>
                    <label>Temporary Nick:
                        <input type="text" ref={this.nickRef} />
                    </label>
                    <input type="submit" value="Enter Happsy" />
                </form>

            </div>
        );
    }
}

export default withRouter(LoginPage);