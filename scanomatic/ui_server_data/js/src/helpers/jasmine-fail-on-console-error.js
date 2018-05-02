beforeEach(() => {
    spyOn(console, 'error').and.callFake((warning) => { throw new Error(warning); });
});
