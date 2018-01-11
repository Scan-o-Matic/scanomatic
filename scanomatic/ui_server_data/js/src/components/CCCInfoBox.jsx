import React from 'react';

import CCCPropTypes from '../prop-types';


export default function CCCInfoBox(props) {
    return (
        <table className="table">
            <thead>
                <tr>
                    <th>Parameters</th>
                    <th>Values</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Id</td><td>{props.cccMetadata.id}</td></tr>
                <tr><td>Token</td><td>{props.cccMetadata.accessToken}</td></tr>
                <tr><td>Species</td><td>{props.cccMetadata.species}</td></tr>
                <tr><td>Reference</td><td>{props.cccMetadata.reference}</td></tr>
                <tr><td>Pinning Format</td><td>{props.cccMetadata.pinningFormat.name}</td></tr>
                <tr><td>Fixture</td><td>{props.cccMetadata.fixtureName}</td></tr>
            </tbody>
        </table>
    );
}

CCCInfoBox.propTypes = {
    cccMetadata: CCCPropTypes.cccMetadata.isRequired,
};
