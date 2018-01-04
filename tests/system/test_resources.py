import requests


class TestCommonResources:

    def test_own_js(self, scanomatic, browser):
        for js_file in (
            'analysis.js',
            'compile.js',
            'ccc.js',
            'experiment.js',
            'fixtures.js',
            'grayscales.js',
            'helpers.js',
            'qc_normAPIHelper.js',
            'qc_normDrawCurves.js',
            'qc_normHelper.js',
            'scanners.js',
            'settings.js',
            'simple_graphs.js',
            'status.js',
        ):
            r = requests.get(scanomatic + '/js/{}'.format(js_file))
            r.raise_for_status()
            assert r.text and len(r.text), '{} is empty'.format(js_file)

    def test_external_js(self, scanomatic, browser):

        for js_file in (
            'bootstrap-toggle.js',
            'bootstrap.js',
            'd3.js',
            'jquery-ui.js',
            'jquery.js',
            'jquery.modal.js',
            'jquery.treetable.js',
            'spin.js',
        ):
            r = requests.get(scanomatic + '/js/external/{}'.format(js_file))
            r.raise_for_status()
            assert r.text and len(r.text), '{} is empty'.format(js_file)

    def test_images(self, scanomatic, browser):

        for im_file in (
            'favicon.ico',
            'stop.png',
            'menu.png',
            'yeastOK.png',
            'yeastNOK.png',
            'scan-o-matic_2.png',
        ):
            r = requests.get(scanomatic + '/images/{}'.format(im_file))
            r.raise_for_status()
            assert r.content and len(r.content), '{} is empty'.format(im_file)

    def test_css(self, scanomatic, browser):

        for css_file in (
            'main.css',
            'qc_norm.css',
        ):
            r = requests.get(scanomatic + '/style/{}'.format(css_file))
            r.raise_for_status()
            assert r.text and len(r.text), '{} is empty'.format(css_file)

    def test_fonts(self, scanomatic, browser):

        for font_file in (
            'glyphicons-halflings-regular.eot',
            'glyphicons-halflings-regular.svg',
            'glyphicons-halflings-regular.ttf',
            'glyphicons-halflings-regular.woff',
            'glyphicons-halflings-regular.woff2',
        ):
            r = requests.get(scanomatic + '/fonts/{}'.format(font_file))
            r.raise_for_status()
            assert r.text and len(r.text), '{} is empty'.format(font_file)
