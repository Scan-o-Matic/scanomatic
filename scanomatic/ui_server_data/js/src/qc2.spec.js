describe('QC2 module', () => {
    const uri = '/somewhere?yes=no no&no=yes';

    describe('getURLParam', () => {
        it('returns nothing if key/name not set', () => {
            expect(window.getURLParam(uri, 'notASetting')).toEqual(null);
        });

        it('returns value for parameter `yes`', () => {
            expect(window.getURLParam(uri, 'yes')).toBe('no no');
        });

        it('returns value for parameter `no`', () => {
            expect(window.getURLParam(uri, 'no')).toBe('yes');
        });
    });

    describe('setQCProjectFromURL', () => {
        let getURLParamSpy;

        describe('not supplying analysis directory', () => {
            beforeEach(() => {
                getURLParamSpy = spyOn(window, 'getURLParam').and.returnValue(null);
            });

            it('Rejects', (done) => {
                window.setQCProjectFromURL().then(() => {}, done);
                expect(getURLParamSpy).toHaveBeenCalled();
            });
        });

        describe('supplying analysis directory', () => {
            let projectSelectionStageSpy;

            beforeEach(() => {
                getURLParamSpy = spyOn(window, 'getURLParam').and.returnValue('silly');
                projectSelectionStageSpy = spyOn(window, 'projectSelectionStage');
                jasmine.Ajax.install();
            });

            afterEach(() => {
                jasmine.Ajax.uninstall();
            });


            it('calls expected end-point', () => {
                window.setQCProjectFromURL();
                expect(jasmine.Ajax.requests.mostRecent().url)
                    .toBe('/api/results/browse/silly');
            });

            it('gets the analysisdirectory parameter from the uri', () => {
                window.setQCProjectFromURL();
                expect(getURLParamSpy).toHaveBeenCalled();
                expect(getURLParamSpy.calls.mostRecent().args[1])
                    .toEqual('analysisdirectory');
            });

            describe('not having analysis', () => {
                let modalMessageSpy;
                beforeEach(() => {
                    modalMessageSpy = spyOn(window, 'modalMessage');
                    jasmine.Ajax
                        .stubRequest('/api/results/browse/silly')
                        .andReturn({ responseText: JSON.stringify({ is_project: false }) });
                });

                it('resolves', (done) => {
                    window.setQCProjectFromURL().then(done);
                });

                it('alerts user to missing analysis', (done) => {
                    window.setQCProjectFromURL().then(() => {
                        expect(modalMessageSpy)
                            .toHaveBeenCalledWith('<strong>Error</strong>: No analysis found!');
                        done();
                    });
                });

                it('sets project selection stage to project', (done) => {
                    window.setQCProjectFromURL().then(() => {
                        expect(projectSelectionStageSpy)
                            .toHaveBeenCalledWith('project');
                        done();
                    });
                });
            });

            describe('having an analysis', () => {
                let waitSpy;
                let fillProjectDetailsSpy;
                const analysisInfo = {
                    is_project: true,
                    project_name: 'nilly',
                };

                beforeEach(() => {
                    waitSpy = spyOn(window, 'wait');
                    fillProjectDetailsSpy = spyOn(window, 'fillProjectDetails');
                    jasmine.Ajax
                        .stubRequest('/api/results/browse/silly')
                        .andReturn({
                            responseText: JSON.stringify(analysisInfo),
                        });
                });

                it('resolves', (done) => {
                    window.setQCProjectFromURL().then(done);
                });

                it('sets project selection stage to project', (done) => {
                    window.setQCProjectFromURL().then(() => {
                        expect(projectSelectionStageSpy)
                            .toHaveBeenCalledWith('project');
                        done();
                    });
                });

                it('activates the wait/talking to server modal', (done) => {
                    window.setQCProjectFromURL().then(() => {
                        expect(waitSpy)
                            .toHaveBeenCalled();
                        done();
                    });
                });

                it('starts processing the project', (done) => {
                    window.setQCProjectFromURL().then(() => {
                        expect(fillProjectDetailsSpy)
                            .toHaveBeenCalledWith(analysisInfo);
                        done();
                    });
                });
            });

            describe('having analysis not supply project_name', () => {
                let fillProjectDetailsSpy;
                const analysisInfo = {
                    is_project: true,
                };

                beforeEach(() => {
                    fillProjectDetailsSpy = spyOn(window, 'fillProjectDetails');
                    spyOn(window, 'wait');
                    jasmine.Ajax
                        .stubRequest('/api/results/browse/silly')
                        .andReturn({
                            responseText: JSON.stringify(analysisInfo),
                        });
                });

                it('gets the project name from the uri', (done) => {
                    window.setQCProjectFromURL().then(() => {
                        expect(getURLParamSpy.calls.mostRecent().args[1])
                            .toEqual('project');
                        done();
                    });
                });

                it('starts processing the project', (done) => {
                    window.setQCProjectFromURL().then(() => {
                        expect(fillProjectDetailsSpy)
                            .toHaveBeenCalledWith({
                                project_name: 'silly',
                                is_project: true,
                            });
                        done();
                    });
                });
            });
        });
    });
});
