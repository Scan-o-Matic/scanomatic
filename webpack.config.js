module.exports = {
    entry: {
        ccc: ['./scanomatic/ui_server_data/js/ccc/index.js'],
    },
    output: {
        filename: 'scanomatic/ui_server_data/js/[name].js',
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['env', 'react']
                    }
                }
            },
            {
                test: /\.png$/,
                use: {
                    loader: 'file-loader',
                }
            },
        ],
    },
};
