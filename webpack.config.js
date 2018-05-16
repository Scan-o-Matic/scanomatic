module.exports = {
    entry: {
        ccc: ['./scanomatic/ui_server_data/js/src/ccc.jsx'],
        scanning: ['./scanomatic/ui_server_data/js/src/scanning.jsx'],
        projects: ['./scanomatic/ui_server_data/js/src/projects'],
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
                },
            },
            {
                test: /\.(png|tiff?)$/,
                use: {
                    loader: 'file-loader',
                },
            },
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader'],
            },
        ],
    },
    resolve: {
        extensions: ['.js', '.json', '.jsx'],
    },
};
