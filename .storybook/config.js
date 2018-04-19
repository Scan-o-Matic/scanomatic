import { configure } from '@storybook/react';

const req = require.context(
    '../scanomatic/ui_server_data/js/src/components',
    true,
    /\.stories\.jsx$/
)

function loadStories() {
  req.keys().forEach((filename) => req(filename))
}

configure(loadStories, module);
