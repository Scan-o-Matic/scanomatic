import Duration from '../src/Duration';

describe('Duration(93784)', () => {
    describe('totalSeconds', () => {
        it('should be 93784', () => {
            const duration = new Duration(93784);
            expect(duration.totalSeconds).toEqual(93784);
        });
    });

    describe('totalMilliseconds', () => {
        it('should be 93784000', () => {
            const duration = new Duration(93784);
            expect(duration.totalMilliseconds).toEqual(93784000);
        });
    });

    describe('days', () => {
        it('should be 1', () => {
            const duration = new Duration(93784);
            expect(duration.days).toEqual(1);
        });
    });

    describe('hours', () => {
        it('should be 2', () => {
            const duration = new Duration(93784);
            expect(duration.hours).toEqual(2);
        });
    });

    describe('minutes', () => {
        it('should be 3', () => {
            const duration = new Duration(93784);
            expect(duration.minutes).toEqual(3);
        });
    });

    describe('seconds', () => {
        it('should be 4', () => {
            const duration = new Duration(93784);
            expect(duration.seconds).toEqual(4);
        });
    });

    describe('after(1985-10-26T01:20:00Z)', () => {
        it('should return 1985-10-27T03:23:04Z', () => {
            const duration = new Duration(93784);
            expect(duration.after(new Date('1985-10-26T01:20:00Z')))
                .toEqual(new Date('1985-10-27T03:23:04Z'));
        });
    });

    describe('before(1985-10-26T01:20:00Z)', () => {
        it('should return 1985-10-24T23:16:56Z', () => {
            const duration = new Duration(93784);
            expect(duration.before(new Date('1985-10-26T01:20:00Z')))
                .toEqual(new Date('1985-10-24T23:16:56Z'));
        });
    });
});
