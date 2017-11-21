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
                test: /\.(js|jsx)$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
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
    resolve: {
        extensions: ['.js', '.json', '.jsx'],
    },
};
