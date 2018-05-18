import { connect } from 'react-redux';

import StatusRoot from '../components/StatusRoot';
import { getScanners, getExperiments, hasLoadedScannersAndExperiments } from '../statuspage/selectors';

function mapStateToProps(state) {
    return {
        scanners: getScanners(state),
        experiments: getExperiments(state),
        hasLoaded: hasLoadedScannersAndExperiments(state),
    };
}

export default connect(mapStateToProps, {})(StatusRoot);
