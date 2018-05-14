import React from 'react';
import PropTypes from 'prop-types';

import myTypes from '../prop-types';
import FixtureImage from './FixtureImage';

export default function FixturesRoot({
    scanners, scannerId, onSelectScanner, onScanOneImage, imageData, imageActions,
}) {
    const contents = [];
    if (scanners.length === 0) {
        contents.push((
            <div className="alert alert-danger" role="alert">
                <strong>Warning!</strong> No scanner is connected to the system.
            </div>
        ));
    } else {
        contents.push((
            <div className="row">
                <div className="col-md-6">
                    <div className="input-group">
                        <span className="input-group-addon">Scanner</span>
                        <select className="form-control" value={scannerId} onChange={e => onSelectScanner(e.target.value)}>
                            <option key="" value="" disabled>Select Scanner</option>
                            {scanners.map(s => (
                                <option key={s.identifier} value={s.identifier}>
                                    {s.name}
                                </option>))}
                        </select>
                    </div>
                </div>
            </div>
        ));
        if (scannerId && scannerId !== '') {
            const scanner = scanners.filter(s => s.identifier === scannerId)[0];
            if (scanner && scanner.power && !scanner.owned) {
                contents.push((
                    <div className="row">
                        <div className="col-md-6">
                            <div className="btn-group" role="group">
                                <button type="button" className="btn btn-default" onClick={onScanOneImage}>Record Image</button>
                            </div>
                        </div>
                    </div>
                ));
            } else if (!scanner) {
                contents.push((
                    <div className="alert alert-danger" role="alert">
                        <strong>Error!</strong> Lost the scanner or are you trying to hack me?
                    </div>
                ));
            } else if (!scanner.power) {
                contents.push((
                    <div className="alert alert-danger" role="alert">
                        <strong>Error!</strong> Scanner is offline.
                    </div>
                ));
            } else {
                contents.push((
                    <div className="alert alert-warning" role="alert">
                        Scanner is occupied. Please try again after its current job is completed.
                    </div>
                ));
            }
        }
        if (imageData) {
            contents.push(<FixtureImage {...imageData} {...imageActions} />);
        }
    }
    return (
        <div>
            <h1>Fixtures</h1>
            {contents}
        </div>
    );
}

FixturesRoot.propTypes = {
    scannerId: PropTypes.string,
    scanners: PropTypes.arrayOf(PropTypes.shape(myTypes.scannerShape)),
    onSelectScanner: PropTypes.func.isRequired,
    onScanOneImage: PropTypes.func.isRequired,
    imageData: PropTypes.shape({
        uri: PropTypes.string.isRequired,
        grayScaleType: PropTypes.string.isRequired,
        grayScale: PropTypes.shape({
            x: PropTypes.arrayOf(PropTypes.number),
            y: PropTypes.arrayOf(PropTypes.number),
            valid: PropTypes.bool.isRequired,
        }),
        plates: PropTypes.arrayOf(PropTypes.shape({
            x1: PropTypes.number.isRequired,
            y1: PropTypes.number.isRequired,
            x2: PropTypes.number.isRequired,
            y2: PropTypes.number.isRequired,
        })),
    }),
    imageActions: PropTypes.shape({
        onChange: PropTypes.func.isRequired,
        onReset: PropTypes.func.isRequired,
        onSave: PropTypes.func.isRequired,
    }).isRequired,
};

FixturesRoot.defaultProps = {
    scannerId: '',
    scanners: [],
    imageData: null,
};
