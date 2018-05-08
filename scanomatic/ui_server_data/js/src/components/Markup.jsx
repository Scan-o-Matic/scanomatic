import PropTypes from 'prop-types';
import React from 'react';

const md = require('markdown-it')({
    linkify: true,
    breaks: true,
    xhtmlOut: true,
})
    .use(require('markdown-it-sup'))
    .use(require('markdown-it-sub'));

export default function Markup({ markdown }) {
    return (<div
        className="markup"
        dangerouslySetInnerHTML={{__html:md.renderInline(markdown) }}
    />);
}

Markup.propTypes = {
    markdown: PropTypes.string.isRequired,
};
