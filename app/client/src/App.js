import React from 'react';
import { BrowserRouter as Router, Route } from "react-router-dom";
import Login from './pages/login';
import Chat from './pages/chat';
import { BodyWrapper, MiddleWrapper } from './components/styled';


function App() {
  return (
    <BodyWrapper>
        <Router>
          <MiddleWrapper>
            <Route path="/" exact component={Login} />
            <Route path="/chat" component={Chat} />
          </MiddleWrapper>
        </Router>
    </BodyWrapper>
  );
}

export default App;
