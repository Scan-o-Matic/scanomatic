export default class GlobalEventsSpy {
    constructor() {
        this.listeners = new Map();
        this.setSpies();
    }

    setSpies() {
        spyOn(document, 'addEventListener').and
            .callFake((evt, callback) => {
                if (this.listeners.has(evt)) throw new Error(`No support for more than one listener for ${evt}`);
                this.listeners.set(evt, callback);
            });
        spyOn(document, 'removeEventListener').and
            .callFake((evt, callback) => {
                if (this.listeners.has(evt) && this.listeners.get(evt) === callback) {
                    this.listeners.delete(evt);
                } else {
                    throw new Error(`${evt} with callback ${callback} not registered`);
                }
            });
    }

    hasEvents(events) {
        return events.every(e => this.listeners.has(e));
    }

    get size() {
        return this.listeners.size;
    }

    simulate(evt, data) {
        if (this.listeners.has(evt)) {
            this.listeners.get(evt)(data);
        } else {
            throw new Error(`${evt} not registered`);
        }
    }
}
