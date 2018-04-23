const webpackConfig = require('./webpack.config');

module.exports = function (config) {
    config.set({
    // base path that will be used to resolve all patterns (eg. files, exclude)
        basePath: 'scanomatic/ui_server_data/',

    // frameworks to use
    // available frameworks: https://npmjs.org/browse/keyword/karma-adapter
        frameworks: ['jasmine'],

    // list of files / patterns to load in the browser
        files: [
            'js/external/jquery.js',
            'js/external/d3.js',
            'js/external/jquery.modal.js',
            'js/external/jquery.treetable.js',
            'js/external/jquery-ui.js',
            'js/qc2.js',
            'js/qc_normHelper.js',
            { pattern: 'js/src/**/*.spec.@(js|jsx)', watched: false },
        ],

        // list of files to exclude
        exclude: [
            'js/image.js',
            '**/*.swp',
            'js/ccc.js',
            'js/scanning.js',
        ],

    // preprocess matching files before serving them to the browser
    // available preprocessors: https://npmjs.org/browse/keyword/karma-preprocessor
        preprocessors: {
            'js/src/**/*.spec.@(js|jsx)': ['webpack'],
        },

    // test results reporter to use
    // possible values: 'dots', 'progress'
    // available reporters: https://npmjs.org/browse/keyword/karma-reporter
        reporters: ['mocha', 'coverage'],

        mochaReporter: {
            ignoreSkipped: true,
        },

        coverageReporter: {
            dir: 'coverage',
            reporters: [
            { type: 'html', subdir: 'report-html' },
            { type: 'lcov', subdir: 'report-lcov' },
            ],
        },


    // web server port
        port: 9876,

    // enable / disable colors in the output (reporters and logs)
        colors: true,

    // level of logging
    // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
        logLevel: config.LOG_INFO,

    // enable / disable watching file and executing tests whenever any file changes
        autoWatch: true,

    // start these browsers
    // available browser launchers: https://npmjs.org/browse/keyword/karma-launcher
        browsers: ['ChromeHeadless'],

        browserNoActivityTimeout: 30000,

    // Continuous Integration mode
    // if true, Karma captures browsers, runs the tests and exits
        singleRun: false,

    // Concurrency level
    // how many browser should be started simultaneous
        concurrency: Infinity,

        webpack: webpackConfig,
    });
};
