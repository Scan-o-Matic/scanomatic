export default class FakePromise {
    then(success, failure) {
        if ('value' in this && success) {
            success(this.value);
        } else if ('error' in this && failure) {
            failure(this.error);
        }
        return this;
    }

    catch(failure) {
        return this.then(null, failure);
    }

    static resolve(value) {
        const fake = new FakePromise();
        fake.value = value;
        return fake;
    }

    static reject(error) {
        const fake = new FakePromise();
        fake.error = error;
        return fake;
    }
}

