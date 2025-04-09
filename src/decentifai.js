// import { Doc, WebrtcProvider, awarenessProtocol } from "https://prafulb.github.io/bundledYjs/dist/yjs-bundle.esm.js"
import { Doc } from "https://esm.sh/yjs"
import { WebrtcProvider } from "https://prafulb.github.io/bundledYjs/y-webrtc.js"

const DEFAULTS = {
    appBasePath: "https://prafulb.github.io/decentifai",
    iceServers: [
        {
            urls: 'stun:stun.l.google.com:19302',
            credsRequired: false
        },
        {
            urls: "turn:turn.speed.cloudflare.com:50000",
            credsRequired: true,
            getCredsFrom: "https://speed.cloudflare.com/turn-creds"
        }],
    maxConns: 20
}

export class Decentifai {
    /**
     * Create a new Decentifai instance, a single entity to manage all P2P connections and learning.
     * @param {Object} options - Configuration options for Decentifai
     * @param {string} options.roomId - Unique identifier for the federation
     * @param {string} options.backend - Framework backend for the model (only supports 'tfjs', 'onnx' and 'generic', but 'onnx' support is not natively available yet)
     * @param {Object} options.model - The actual model in the case of TF.js; an object containing paths to the checkpoint state and the training, optimizer and evaluation models in the case of ONNX; or an object containing train, test, extractParameters and updateParameters functions.
     * @param {Function} options.model.train - Function to train the model. Optional in case of a TF.js model where trainingData is supplied.
     * @param {Function} options.model.test - Function to test the model.
     * @param {Function} options.model.extractLocalParametersFunc - Function to extract parameters from the model (default provided for TF.js models)
     * @param {Function} options.model.updateLocalParametersFunc - Function to update model with new parameters (default provided for TF.js models)
     * @param {Object|Array|Function} options.trainingData - Local training dataset, or a function that returns training data for a given round
     * @param {Object|Array|Function} options.testData - Local test dataset, or a function that returns test data
     * @param {Object} options.trainingOptions - Options to pass to the model during training
     * @param {Object} options.federationOptions - Options corresponding to how the federation should be conducted.
     * @param {String} options.federationOptions.password - Password that would be required to join the federation.
     * @param {String} options.federationOptions.aggregationMethod - Aggregation strategy to be used for parameter aggregation between rounds (optional, defaults to 'federatedAveraging').
     * @param {Function} options.federationOptions.aggregateParametersFunc - Custom aggregation function to be used in place of aggregationMethod.
     * @param {Number} options.federationOptions.minPeers - Minimum number of connected peers to reach quorum
     * @param {Number} options.federationOptions.maxPeers - Maximum number of connected peers to be allowed in the federation.
     * @param {Number} options.federationOptions.minRounds - Minimum training rounds to run the federation for.
     * @param {Number} options.federationOptions.maxRounds - Maximum training rounds to run the federation for.
     * @param {Number} options.federationOptions.waitTime - Wait time (in milliseconds) between federation actions, in order to account for network asynchronicity between peers.
     * @param {Object} options.federationOptions.convergenceThresholds - Thresholds to check for model convergence.
     * @param {Number} options.federationOptions.convergenceThresholds.parameterDistance - RMS distance of model parameters across stability window before convergence
     * @param {Number} options.federationOptions.convergenceThresholds.lossDelta - Change in loss across stability window before convergence should be claimed
     * @param {Number} options.federationOptions.convergenceThresholds.accuracyDelta - Change in accuracy across stability window before convergence can be claimed
     * @param {Number} options.federationOptions.convergenceThresholds.stabilityWindow - Window size of consecutive rounds to check if convergence thresholds are met
     * @param {boolean} options.autoTrain - Whether to automatically manage training rounds
     * @param {Array} options.signaling - Array of signaling server URLs
     * @param {Object} options.metadata - Additional metadata about this peer
     * @param {Array} options.iceServers - ICE servers to be used for WebRTC connections.
     * @param {boolean} options.debug - Enable debug logging
     *
     * @fires Decentifai#peersAdded - Fired when new peers connect to the federation
     * @fires Decentifai#peersChanged - Fired when peer information is updated
     * @fires Decentifai#peersRemoved - Fired when peers disconnect from the federation
     * @fires Decentifai#parametersReceived - Fired when parameters are received from other peers
     * @fires Decentifai#parametersShared - Fired when the local peer shares its parameters
     * @fires Decentifai#parametersApplied - Fired when new parameters are applied to the local model
     * @fires Decentifai#roundChanged - Fired when the current training round changes
     * @fires Decentifai#roundProposed - Fired when a new training round is proposed
     * @fires Decentifai#roundQuorumReached - Fired when enough peers have acknowledged a round proposal
     * @fires Decentifai#roundStarted - Fired when a training round starts
     * @fires Decentifai#localTrainingCompleted - Fired when local training completes for a round
     * @fires Decentifai#roundFinalized - Fired when a training round is finalized and parameters are aggregated
     * @fires Decentifai#autoTrainingStarted - Fired when automatic training begins
     * @fires Decentifai#autoTrainingRoundCompleted - Fired when an automatic training round completes
     * @fires Decentifai#autoTrainingStopped - Fired when automatic training stops
     * @fires Decentifai#autoTrainingPaused - Fired when automatic training is paused
     * @fires Decentifai#autoTrainingError - Fired when an error occurs during automatic training
     * @fires Decentifai#modelConverged - Fired when the model has converged
     * @fires Decentifai#disconnected - Fired when disconnecting from the federated learning network
     */
    constructor(options) {
        this.options = {
            roomId: 'Decentifai-test',
            signaling: ['wss://signalyjs-df59a68bd6e6.herokuapp.com'],
            debug: false,
            backend: 'tfjs', // Default to tfjs
            autoTrain: false, // Do not start automatically by default
            iceServers: [],
            ...options,
            metadata: {
                ...options.metadata
            },
            trainingOptions: {
                ...options.trainingOptions
            },
            testOptions: {
                ...options.testOptions
            },
            federationOptions: {
                password: "",
                aggregationMethod: 'federatedAveraging',
                minPeers: 2,
                maxPeers: 5,
                minRounds: 10,
                maxRounds: 100,
                waitTime: 2000,
                ...options.federationOptions,
                convergenceThresholds: {
                    parameterDistance: 0.001,
                    lossDelta: 0.01,
                    accuracyDelta: 0.01,
                    maxRounds: 50,
                    stabilityWindow: 2,
                    ...options.federationOptions?.convergenceThresholds
                }
            },
        }

        if (!this.options.model) {
            throw new Error('A model object must be provided.')
        }

        if (!['tfjs', 'onnx', 'generic'].includes(this.options.backend)) {
            throw new Error('Backend must be one of "tfjs", "onnx" or "generic"')
        }

        this.onDeviceTraining = this.options.backend !== 'generic'

        this.model = this.options.model
        // Set up default parameter functions based on backend
        if (!this.model.train || typeof (this.model.train) !== 'function') {
            if (!this.onDeviceTraining) {
                throw new Error('Function to Train Model must be specified for generic models.')
            }
            this._getDefaultModelTrainingFunction(this.options.backend).then(func => {
                this.model.train = func
            })
        }
        if (!this.model.extractLocalParametersFunc || typeof (this.this.model.extractLocalParametersFunc) !== 'function') {
            if (!this.onDeviceTraining) {
                throw new Error('Function to Extract Local Parameters must be specified for generic models.')
            }
            this._getDefaultExtractFunction(this.options.backend).then(func => {
                this.model.extractLocalParametersFunc = func
            })
        }

        if (!this.model.updateLocalParametersFunc || typeof (this.this.model.updateLocalParametersFunc) !== 'function') {
            if (!this.onDeviceTraining) {
                throw new Error('Function to Update Local Parameters must be specified for generic models.')
            }
            this._getDefaultUpdateFunction(this.options.backend).then(func => {
                this.model.updateLocalParametersFunc = func
            })
        }

        this.peers = {}
        this.trainingRound = 0
        this.events = new EventTarget()
        this.autoTrainingEnabled = !!this.options.autoTrain
        this.trainingData = this.options.trainingData || null
        this.isTraining = false
        this.converged = false

        if (!this.options.federationOptions.aggregateParametersFunc || typeof (this.options.federationOptions.aggregateParametersFunc) !== 'function') {
            if (this.options.federationOptions.aggregationMethod) {
                this._getAggregator(this.options.federationOptions.aggregationMethod).then(func => {
                    this.options.federationOptions.aggregateParametersFunc = func
                })
            } else {
                throw new Error('Aggregation Method or function implementation missing.')
            }
        }
        this.convergenceHistory = []
        this.convergenceMetrics = {
            parameterDistance: [],
            modelLoss: [],
            trainingAccuracy: []
        }
        this.convergenceThresholds = this.options.federationOptions.convergenceThresholds

        this._initYDoc()
        this._setupWebRTC().then(() => {
            this.log('Federated Learning instance initialized with backend:', this.options.backend)

            if (this.autoTrainingEnabled) {
                this._setupAutoTraining()
                this.log('Auto-training mode enabled')
            }
        })


    }

    /**
     * Get default parameter extraction function based on backend
     * @private
     * @param {string} backend - ML backend type ('tfjs' or 'onnx')
     * @returns {Function} - Parameter extraction function
     */
    async _getDefaultExtractFunction(backend) {
        const { extractLocalParameters } = await import(`${DEFAULTS.appBasePath}/utils/extractors.js`)
        return extractLocalParameters(backend)
    }

    /**
     * Get default parameter update function based on backend
     * @private
     * @param {string} backend - ML backend type ('tfjs' or 'onnx')
     * @returns {Function} - Parameter update function
     */
    async _getDefaultUpdateFunction(backend) {
        const { updateLocalParameters } = await import(`${DEFAULTS.appBasePath}/utils/updators.js`)
        return updateLocalParameters(backend)
    }

    /**
     * Get default parameter aggregation function (Federated Averaging)
     * @private
     * @returns {Function} - Parameter aggregation function
     */
    async _getAggregator(aggregationMethod) {
        const aggregators = await import(`${DEFAULTS.appBasePath}/utils/aggregators.js`)
        return aggregators[aggregationMethod]
    }

    /**
     * Initialize the Y.js document
     * @private
     */
    _initYDoc() {
        this.ydoc = new Doc()

        // Create shared data structures
        this.parameters = this.ydoc.getMap('parameters')
        this.metadata = this.ydoc.getMap('metadata')
        this.roundInfo = this.ydoc.getMap('roundInfo')

        // Initialize metadata
        this.metadata.set('peerId', this.ydoc.clientID)

        // Set user-provided metadata
        Object.entries(this.options.metadata).forEach(([key, value]) => {
            this.metadata.set(key, value)
        })

        this.log('Y.js document initialized')
    }

    /**
     * Set up WebRTC connections using y-webrtc
     * @private
     */

    async _setupWebRTC() {
        let iceServers = this.options.iceServers
        if (iceServers.length === 0) {
            for (const iceServer of DEFAULTS.iceServers) {
                if (iceServer.credsRequired) {
                    const { username, credential } = await (await fetch(iceServer.getCredsFrom)).json()
                    iceServer.username = username
                    iceServer.credential = credential
                }
                iceServers.push(iceServer)
            }
        }

        const webrtcProviderOptions = {
            signaling: this.options.signaling,
            password: this.options.password,
            // awareness: new awarenessProtocol.Awareness(this.ydoc),
            maxConns: DEFAULTS.maxConns,
            peerOpts: {
                config: {
                    iceServers
                }
            }
        }

        this.provider = new WebrtcProvider(this.options.roomId, this.ydoc, webrtcProviderOptions)

        this.awareness = this.provider.awareness
        console.log(this.awareness)

        // Set up awareness
        this.awareness.setLocalState({
            clientID: this.ydoc.clientID,
            lastSeen: Date.now(),
            online: true,
            training: false,
            round: this.trainingRound,
            metadata: this.options.metadata
        })

        // Handle peer connections

        this.provider.on('peers', (event) => {
            this._handlePeerUpdate(event)
        })
        this.awareness.on('update', (event) => {
            this._handlePeerUpdate(event)
        })

        // Handle parameter updates
        this.parameters.observe(event => {
            this._handleParameterUpdate(event)
        })

        // Handle round info updates
        this.roundInfo.observe(event => {
            this._handleRoundUpdate(event)
        })

        this.log('WebRTC provider initialized with room:', this.options.roomId)
        this.log(`Self ID [${this.ydoc.clientID}] Available to connect with peers`)
    }

    /**
     * Get WebRTC peer ID of current instance as seen by others
     */
    getSelfPeerId() {
        return this.ydoc.clientID
    }

    /**
     * Handle peer connection updates
     * @private
     * @param {Object} event - Peer update event
     */
    _handlePeerUpdate(event) {
        const { added, updated, removed } = event

        added?.forEach(clientID => {
            const state = this.awareness.getStates().get(clientID)
            const peerExists = !!this.peers[clientID]

            if (state?.online) {
                this.peers[clientID] = {
                    clientID,
                    connected: true,
                    lastSeen: Date.now(),
                    metadata: state.metadata
                }
                if (!peerExists) {
                    this.log(`New peer ${state?.metadata?.name} connected: ${clientID}`)
                    this._dispatchEvent('peersAdded', { peers: Object.keys(this.peers) })
                }
            }
        })

        updated?.forEach(clientID => {
            const state = this.awareness.getStates().get(clientID)
            if (state) {
                const peer = this.peers[clientID]
                if (peer) {
                    peer.lastSeen = Date.now()
                    peer.metadata = state.metadata
                    this.peers[clientID] = peer
                }
                this._dispatchEvent('peersChanged', { peers: Object.keys(this.peers) })
                // this.log(`Peer ${state.metadata?.name} updated via awareness.`)
            }
        })

        removed?.forEach(clientID => {
            if (!!this.peers[clientID]) {
                const peer = this.peers[clientID]
                this.log(`Peer ${peer.metadata?.name} disconnected via awareness: ${clientID}`)
                delete this.peers[clientID]
                this._dispatchEvent('peersRemoved', { peers: Object.keys(this.peers) })
            }
        })

    }

    /**
     * Handle parameter updates from peers
     * @private
     * @param {Object} event - Parameter update event
     */
    _handleParameterUpdate(event) {
        if (this.isTraining) {
            return // Don't process updates while we're training
        }

        // Get the updated parameters
        const updatedParams = {}
        event.changes.keys.forEach((change, key) => {
            const changedParams = this.parameters.get(key)
            if (changedParams.peerId !== this.getSelfPeerId() && change.action === 'add' || change.action === 'update') {
                this.log('Parameter update received')
                updatedParams[key] = changedParams
            }
        })

        if (Object.keys(updatedParams).length > 0) {
            this._dispatchEvent('parametersReceived', {
                parameters: updatedParams,
                source: event.transaction.origin
            })
        }
    }

    /**
     * Handle round info updates
     * @private
     * @param {Object} event - Round update event
     */
    _handleRoundUpdate(event) {
        const roundData = this.roundInfo.get('roundData')

        if (roundData.initiator !== this.getSelfPeerId()) {
            if (roundData?.currentRound === this.trainingRound) {
                if (roundData.status === 'training') {
                    this._startTrainingRound()
                }
                else if (roundData.status === 'completed') {
                    this.finalizeRound()
                }
            }

            if (roundData?.currentRound > this.trainingRound) {
                if (roundData.status === 'proposed') {
                    // Check if this peer is "really behind" in the federation. Forcibly update it to the latest aggregated parameter set if so.
                    if (this.trainingRound < roundData.currentRound - 1) {
                        const peerParameters = this.getParameters(roundData.currentRound - 1)
                        const aggregatedParams = this.options.federationOptions.aggregateParametersFunc(peerParameters, this.options.backend)
                        this.model.updateLocalParametersFunc(aggregatedParams)
                    }
                    this.trainingRound = roundData.currentRound

                    // Acknowledge the round proposal via awareness
                    const awareness = this.awareness.getLocalState() || {}
                    awareness.round = this.trainingRound
                    awareness.roundStatus = 'acknowledged'
                    this.awareness.setLocalState(awareness)
                }
            }

            this._dispatchEvent('roundChanged', { round: this.trainingRound })
        }
    }

    /**
     * Set up auto-training event listeners and initialization
     * @private
     */
    _setupAutoTraining() {
        if (!this.trainingData && this.onDeviceTraining) {
            this.log('Warning: Auto-training enabled but no training data provided')
            return
        }

        // Listen for peer changes to potentially start training
        this.on('peersAdded', () => {
            this._checkShouldStartTraining()
        })

        // Listen for round completion to continue training if needed
        this.on('roundFinalized', () => {
            this._continueTrainingIfNeeded()
        })

        // Listen for model convergence to stop auto-training
        this.on('modelConverged', () => {
            this.converged = true
            this.isTraining = false
            this.log('Auto-training stopped: model converged')
        })
    }

    /**
     * Check if auto-training should begin
     * @private
     */
    _checkShouldStartTraining() {
        if (!this.autoTrainingEnabled || this.isTraining || this.converged) {
            return
        }

        if (1 + Object.keys(this.peers).length >= this.options.federationOptions.minPeers) {
            this._startAutoTraining()
        }
    }

    /**
     * Start the automatic training process
     * @private
     */
    async _startAutoTraining() {
        if (this.isTraining) {
            return
        }

        this.log('Starting auto-training process')
        this._dispatchEvent('autoTrainingStarted', {})

        // Begin the first training round
        await this._runTrainingRound()
    }

    /**
     * Continue training if the model hasn't converged
     * @private
     */
    async _continueTrainingIfNeeded() {
        if (!this.autoTrainingEnabled || this.isTraining || this.converged) {
            return
        }

        // Check if we've reached max rounds
        if (this.trainingRound >= this.convergenceThresholds.maxRounds) {
            this.log('Auto-training stopped: maximum rounds reached')
            this.isTraining = false
            this._dispatchEvent('autoTrainingStopped', { reason: 'maxRoundsReached' })
            return
        }

        // Add a small delay between rounds to allow network synchronization
        await new Promise(resolve => setTimeout(resolve, this.options.federationOptions.waitTime))

        // Check if peers are still connected
        if (1 + Object.keys(this.peers).length < this.options.federationOptions.minPeers) {
            this.log('Auto-training paused: not enough peers')
            this.isTraining = false
            this._dispatchEvent('autoTrainingPaused', { reason: 'insufficientPeers' })
            return
        }

        // Start next training round
        await this._runTrainingRound()
    }

    /**
     * Check how many peers shared parameter updates for the current round
     */
    getNumPeersWhoSharedParameters() {
        const peerParameters = this.getParameters(this.trainingRound)
        return peerParameters.length
    }

    /**
     * Run a single training round in auto-training mode
     * @private
     */
    async _runTrainingRound() {
        if (this.trainingData) {
            try {
                this.log(`Auto-training round ${this.trainingRound + 1}`)
                await new Promise(res => setTimeout(res, Math.random() * this.options.federationOptions.waitTime)) // Random wait to allow a proposer to initiate training and avoid multiple proposals happening simultaneously.

                const proposalQuorumReached = await this.proposeTrainingRound()
                await new Promise(res => setTimeout(res, this.options.federationOptions.waitTime))
                if (proposalQuorumReached) {
                    await this.startTrainingRound()
                } else {
                    this.trainingRound--
                    this.log(`Could not reach quorum for ${this.trainingRound + 1}. Retrying...`)
                    return
                }

                let parameterSharingCountdown = 10
                const enoughParametersShared = await new Promise(res => {
                    const checkIfEnoughParametersShared = setInterval(() => {
                        parameterSharingCountdown--
                        const numPeersWhoSharedParameters = this.getNumPeersWhoSharedParameters()
                        if (numPeersWhoSharedParameters >= this.options.federationOptions.minPeers) {
                            clearInterval(checkIfEnoughParametersShared)
                            res(true)
                        } else if (parameterSharingCountdown === 0) {
                            clearInterval(checkIfEnoughParametersShared)
                            res(false)
                        }
                    }, this.options.federationOptions.waitTime)
                })

                if (enoughParametersShared) {
                    await this.finalizeRound()
                    await new Promise(res => setTimeout(res, this.options.federationOptions.waitTime))

                    this._dispatchEvent('autoTrainingRoundCompleted', {
                        round: this.trainingRound,
                        isConverged: this.converged
                    })
                } else {
                    this.log(`Parameters missing from peers, could not reach quorum for aggregation. Moving on to the next round...`)
                    return
                }

            } catch (error) {
                this.log('Error in auto-training round:', error)
                this.isTraining = false
                this._dispatchEvent('autoTrainingError', { error: error.message })
            }
        }
    }

    /**
     * Enable or disable automatic training
     * @param {boolean} enable - Whether to enable auto-training
     * @param {Object|Array|Function} trainingData - Training data to use (optional)
     * @param {Object} trainingOptions - Options for training (optional)
     */
    setAutoTraining(enable, trainingData = null, trainingOptions = null) {
        this.autoTrainingEnabled = !!enable

        if (trainingData !== null) {
            this.trainingData = trainingData
        }

        if (trainingOptions !== null) {
            this.options.trainingOptions = trainingOptions
        }

        if (this.autoTrainingEnabled) {
            this._setupAutoTraining()
            this._checkShouldStartTraining()
        } else {
            this.isTraining = false
        }

        this.log(`Auto-training ${this.autoTrainingEnabled ? 'enabled' : 'disabled'}`)

        return this.autoTrainingEnabled
    }

    /**
     * Check if training should start or stop based on peer count
     * @private
     */
    _checkTrainingStatus() {
        // For manual training mode
        // if (!this.autoTrainingEnabled && 1 + this.peers.size >= this.options.convergenceThresholds.minPeers && 
        //     !this.isTraining && !this.roundInfo.get('roundData')?.status) {
        //   this.proposeTrainingRound()
        // }

        // For auto training mode - check if we should start/resume
        if (this.autoTrainingEnabled && !this.isTraining && !this.converged) {
            this._checkShouldStartTraining()
        }
    }

    /**
     * Check if training should start or stop based on peer count
     * @
     */
    checkTrainingStatus() {
        // For manual training mode
        if (!this.autoTrainingEnabled && 1 + Object.keys(this.peers).length >= this.options.federationOptions.minPeers &&
            !this.isTraining) {
            return true
        }
    }

    /**
     * Train the local model on local data and share parameters with peers afterwards.
     * @param {Array|Object} data - Training data
     * @param {Object} options - Training options to pass to the model's `fit` function
     * @returns {Promise} - Resolves with model training output when training is complete
     */
    async trainLocal(data, options = {}) {
        if (!data && this.onDeviceTraining) {
            throw new Error('Training data must be provided')
        }
        this.log('Starting local training')
        this.isTraining = true

        try {
            // Update awareness state
            this.awareness.setLocalState({
                ...this.awareness.getLocalState(),
                training: this.isTraining
            })

            let info
            if (typeof this.model.train === 'function') {
                info = await this.model.train({ data, options })
            } else {
                throw new Error('Model must have a train method')
            }

            this.log('Local training completed for current round')

            const params = this.model.extractLocalParametersFunc(this.model)
            this.log('Sharing parameters for round', this.trainingRound)
            this.shareParameters(params)

            return info
        } catch (error) {
            this.warn('Training error:', error)
            throw error
        } finally {
            this.isTraining = false
            this.awareness.setLocalState({
                ...this.awareness.getLocalState(),
                training: this.isTraining
            })
        }
    }

    /**
     * Share local model parameters with peers
     * @param {Object} parameters - The model parameters to share
     */
    shareParameters(parameters) {
        if (!parameters) {
            parameters = this.model.extractLocalParametersFunc(this.model)
        }

        this.log('Sharing parameters with peers')

        // Add metadata to parameters
        const paramUpdate = {
            peerId: this.ydoc.clientID,
            timestamp: Date.now(),
            round: this.trainingRound,
            parameters: parameters
        }

        // Update Y.js shared document
        this.parameters.set(`peer_${this.ydoc.clientID}`, paramUpdate)

        this._dispatchEvent('parametersShared', { parameters: paramUpdate })
    }

    /**
     * Propose a new training round to all peers.
     * @returns {boolean} - Flag specifying whether quorum was reached in the specified wait time.
     */
    async proposeTrainingRound() {
        // Check if round has already been proposed.
        const roundData = this.roundInfo.get('roundData')
        if (roundData?.currentRound === this.trainingRound && ['proposed', 'training'].includes(roundData?.status)) {
            return true
        }

        this.trainingRound++

        this.log(`Proposing training round ${this.trainingRound}`)

        // Update roundInfo in Y.js document
        this.roundInfo.set('roundData', {
            currentRound: this.trainingRound,
            status: 'proposed', // New status before 'training'
            initiator: this.ydoc.clientID,
            proposeTime: Date.now()
        })

        // Update local awareness
        const awareness = this.awareness.getLocalState() || {}
        awareness.round = this.trainingRound
        awareness.roundStatus = 'acknowledged' // Mark as acknowledged by proposer
        this.awareness.setLocalState(awareness)

        this._dispatchEvent('roundProposed', { round: this.trainingRound })
        let quorumCheckCountdown = 10 // PARAMETER!!! Should be defined by user ideally.
        // Wait for acknowledgments before starting training
        const quorumReached = await new Promise((res, _) => {

            const checkForQuorum = setInterval(() => {
                quorumCheckCountdown--
                if (this._checkRoundAcknowledgments()) {
                    clearInterval(checkForQuorum)
                    res(true)
                } else if (quorumCheckCountdown === 0) {
                    clearInterval(checkForQuorum)
                    res(false)
                }
            }, this.options.federationOptions.waitTime)
        })

        if (quorumReached) {
            this._dispatchEvent('roundQuorumReached', {
                round: this.trainingRound
            })
        }

        return quorumReached

    }

    /**
     * Check if enough peers acknowledged the round
     */
    _checkRoundAcknowledgments() {

        let acknowledgedPeers = 0
        this.awareness.getStates().forEach((state, clientId) => {
            if (clientId !== this.ydoc.clientID && state.online && state.round === this.trainingRound && state.roundStatus === 'acknowledged') {
                acknowledgedPeers++
            }
        })

        const minRequired = this.options.federationOptions.minPeers - 1
        if (acknowledgedPeers >= minRequired) {
            this.log(`Round ${this.trainingRound} acknowledged by ${acknowledgedPeers} peers, ready to start training`)
            return true
        } else {
            this.log(`Timeout waiting for round acknowledgments. Only ${acknowledgedPeers}/${Object.keys(this.peers).length} peers acknowledged.`)
            return false
        }
    }

    /**
     * Start a training round
     * @private
     */

    async _startTrainingRound() {
        this.log(`Starting training round ${this.trainingRound}`)

        this._dispatchEvent('roundStarted', { round: this.trainingRound })

        return await this._startLocalTrainingRound()
    }

    /**
     * Start a training round
     * @returns {modelInfo} - Model metrics for the current training round
     */
    async startTrainingRound() {
        if (this.isTraining) {
            return
        }

        this.roundInfo.set('roundData', {
            ...this.roundInfo.get('roundData'),
            status: 'training',
            startTime: Date.now()
        })
        return await this._startTrainingRound()
    }

    /**
     * Start a local training round
     * @private
     */
    async _startLocalTrainingRound() {
        const data = typeof this.trainingData === 'function'
            ? await this.trainingData(this.trainingRound + 1)
            : this.trainingData

        if (!data && this.onDeviceTraining && (!this.model.train || typeof (this.model.train) !== 'function')) {
            throw new Error('No training data available')
        }
        const modelInfo = await this.trainLocal(data, this.options.trainingOptions)
        this._dispatchEvent('localTrainingCompleted', {
            round: this.trainingRound,
            isConverged: this.converged,
            modelInfo
        })
        return modelInfo
    }

    // /**
    //  * Test model on provided test dataset
    //  * @private
    //  */
    // async _testModel() {
    //     if (this.model.test) {
    //         const predictions = await model.test(data, {
    //             batchSize
    //         })
    //         const testLabels = data.y
    //         this._calculateMetrics(predictions, labels)
    //     }
    // }

    // _calculateMetrics(predictedLabels, groundTruth) {

    //     console.log("Predictions: ", predictions)
    //     console.log("Actual Labels: ", groundTruth)
    //     const numCorrectPredictions = predictions.reduce((correctPreds, prediction, index) => {
    //     // console.log(`Predicted vs Actual Labels for Test Observation ${index + 1} : ${[prediction, groundTruth[index]]}`)
    //     if (prediction === groundTruth[index]) {
    //         correctPreds += 1
    //     }
    //     return correctPreds
    //     }, 0)
    //     console.log(`Test Accuracy: ${100 * numCorrectPredictions/groundTruth.length}`)
    // }

    /**
     * Calculate parameter distance between two parameter sets
     * @param {Object} params1 - First set of parameters
     * @param {Object} params2 - Second set of parameters
     * @returns {number} - Euclidean distance between parameters
     */
    _calculateDistance(params1, params2) {
        const keys = Object.keys(params1)
        let squaredDiffSum = 0

        keys.forEach(key => {
            if (typeof params1[key] === 'number' && typeof params2[key] === 'number') {
                const diff = params1[key] - params2[key]
                squaredDiffSum += diff * diff
            }
            // For TF.js tensors
            else if (Array.isArray(params1[key][0].values) && Array.isArray(params2[key][0].values)) {
                squaredDiffSum = params1[key][0].values.reduce((sum, value, index) => {
                    const difference = value - params2[key][0].values[index]
                    return sum + difference * difference
                }, 0)
            }
            // Needs additional check for ONNX tensors
        })
        if (typeof (squaredDiffSum) === 'number') {
            return Math.sqrt(squaredDiffSum)
        } else if (Array.isArray(squaredDiffSum)) {
            return
        }
    }

    /**
     * Check if the model has converged
     * @returns {boolean} - Whether the model has converged
     */
    _checkConvergence() {
        const { parameterDistance, modelLoss, trainingAccuracy } = this.convergenceMetrics

        // Convergence check based on multiple metrics
        if (parameterDistance.length < this.convergenceThresholds.stabilityWindow) {
            return false
        }

        // Check parameter distance stability
        const recentDistances = parameterDistance.slice(-this.convergenceThresholds.stabilityWindow)
        const distanceStable = recentDistances.every(
            (dist, i) => i === 0 || Math.abs(dist - recentDistances[i - 1]) < this.convergenceThresholds.parameterDistance
        )

        // Check loss delta stability
        const recentLosses = modelLoss.slice(-this.convergenceThresholds.stabilityWindow)
        const lossesStable = recentLosses.every(
            (dist, i) => i === 0 || Math.abs(dist - recentLosses[i - 1]) < this.convergenceThresholds.lossDelta
        )

        // Check accuracy delta stability
        const recentAccuracies = trainingAccuracy.slice(-this.convergenceThresholds.stabilityWindow)
        const accuracyStable = recentAccuracies.every(
            (dist, i) => i === 0 || Math.abs(dist - recentAccuracies[i - 1]) < this.convergenceThresholds.accuracyDelta
        )

        // Additional convergence criteria can be added here
        const roundsWithinLimit = this.trainingRound < this.convergenceThresholds.maxRounds

        return distanceStable && lossesStable && accuracyStable && roundsWithinLimit
    }

    _trackConvergenceMetrics() {
        // Collect parameters from current round
        let currentParams, currentLoss, currentAccuracy

        // Compare with previous parameters if available
        currentParams = this.model.extractLocalParametersFunc(this.model)
        if (this.convergenceHistory.length > 0) {
            const previousParams = this.convergenceHistory[this.convergenceHistory.length - 1].modelParameters
            const paramDistance = this._calculateDistance(currentParams, previousParams)
            this.convergenceMetrics.parameterDistance.push(paramDistance)
        }

        if (typeof this.model.getLoss === 'function') {
            currentLoss = this.model.getLoss()
            if (this.convergenceHistory.length > 0) {
                const previousLoss = this.convergenceHistory[this.convergenceHistory.length - 1].modelLoss
                const lossDelta = this._calculateDistance(currentLoss, previousLoss)
                this.convergenceMetrics.modelLoss.push(lossDelta)
            }
        }

        if (typeof this.model.getAccuracy === 'function') {
            currentAccuracy = this.model.getAccuracy()
            if (this.convergenceHistory.length > 0) {
                const previousAccuracy = this.convergenceHistory[this.convergenceHistory.length - 1].trainingAccuracy
                const accuracyDelta = this._calculateDistance(currentAccuracy, previousAccuracy)
                this.convergenceMetrics.trainingAccuracy.push(accuracyDelta)
            }
        }

        // Store current parameters
        this.convergenceHistory.push({
            modelParameters: currentParams,
            modelLoss: currentLoss,
            trainingAccuracy: currentAccuracy,
            round: this.trainingRound
        })
    }

    /**
     * Visualize convergence metrics
     * @returns {Object} - Convergence visualization data
     */
    getConvergenceVisualization() {
        const { parameterDistance, modelLoss, trainingAccuracy } = this.convergenceMetrics

        return {
            rounds: this.convergenceHistory.length,
            parameterDistance: {
                data: parameterDistance,
                average: parameterDistance.length
                    ? parameterDistance.reduce((a, b) => a + b, 0) / parameterDistance.length
                    : 0,
                trend: this._calculateTrend(parameterDistance)
            },
            modelLoss: {
                data: modelLoss,
                average: modelLoss.length
                    ? modelLoss.reduce((a, b) => a + b, 0) / modelLoss.length
                    : 0,
                trend: this._calculateTrend(modelLoss)
            },
            trainingAccuracy: {
                data: trainingAccuracy,
                average: trainingAccuracy.length
                    ? trainingAccuracy.reduce((a, b) => a + b, 0) / trainingAccuracy.length
                    : 0,
                trend: this._calculateTrend(trainingAccuracy)
            }
        }
    }

    /**
     * Calculate trend of a metric series
     * @private
     * @param {Array} series - Numerical series to analyze
     * @returns {string} - Trend description
     */
    _calculateTrend(series) {
        if (series.length < 2) return 'insufficient data'

        const lastDiff = series[series.length - 1] - series[series.length - 2]

        if (Math.abs(lastDiff) < this.convergenceThresholds.parameterDistance) {
            return 'stable'
        } else if (lastDiff > 0) {
            return 'increasing'
        } else {
            return 'decreasing'
        }
    }

    /**
     * Get parameter set from all peers for a given round
     * @returns {Array} - Parameters from every peer for the requested round
     */
    getParameters(round) {
        const peerParameters = []

        this.parameters.forEach((value, key) => {
            if (value.round === round) {
                peerParameters.push(value)
            }
        })
        return peerParameters
    }

    /**
     * Finalize a training round and aggregate parameters
     */
    async finalizeRound() {
        this.log(`Finalizing training round ${this.trainingRound}`)

        const peerParameters = this.getParameters(this.trainingRound)

        if (peerParameters.length >= this.options.federationOptions.minPeers) {
            this.log(`Aggregating parameters from ${peerParameters.length} peers`)

            const aggregatedParams = this.options.federationOptions.aggregateParametersFunc(peerParameters, this.options.backend)

            await this.model.updateLocalParametersFunc(this.model, aggregatedParams)

            const currentRoundData = this.roundInfo.get('roundData')
            if (currentRoundData.initiator === this.getSelfPeerId()) {
                this.roundInfo.set('roundData', {
                    ...this.roundInfo.get('roundData'),
                    currentRound: this.trainingRound,
                    status: 'completed',
                    endTime: Date.now(),
                    participantCount: peerParameters.length
                })
            }

            this._trackConvergenceMetrics()

            // Check if model has converged
            if (this._checkConvergence()) {
                this.log('Model has converged')
                this._dispatchEvent('modelConverged', {
                    round: this.trainingRound,
                    convergenceMetrics: this.convergenceMetrics
                })
            }

            this._dispatchEvent('roundFinalized', {
                round: this.trainingRound,
                participants: peerParameters.length,
                parameters: aggregatedParams
            })

            return true
        } else {
            this.log('Not enough parameters to aggregate yet. Retrying...')
            return false
        }
    }

    /**
     * Apply parameters from peers to the local model
     * @param {Object} parameters - Parameters to apply
     * @returns {Promise} - Resolves when parameters are applied
     */
    async applyParameters(parameters) {
        if (!parameters) {
            throw new Error('Parameters must be provided')
        }

        this.log('Applying parameters to local model')

        try {
            await this.model.updateLocalParametersFunc(this.model, parameters)
            this._dispatchEvent('parametersApplied', { parameters })
            return true
        } catch (error) {
            this.log('Error applying parameters:', error)
            throw error
        }
    }

    /**
     * Reset convergence state to allow restarting training
     */
    resetConvergence() {
        this.converged = false
        this.convergenceHistory = []
        this.convergenceMetrics = {
            parameterDistance: [],
            modelLoss: [],
            trainingAccuracy: []
        }
        this.log('Convergence state reset')
    }

    /**
     * Get the current state of all peers
     * @returns {Array} - Array of peer objects
     */
    getPeers(peerId) {
        if (peerId) {
            return this.peers[peerId]
        }
        const allPeers = Object.values(this.peers)
        return allPeers
    }

    /**
     * Get the current training round
     * @returns {number} - Current training round
     */
    getCurrentRound() {
        return this.trainingRound
    }

    /**
     * Register an event listener
     * @param {string} event - Event name
     * @param {Function} callback - Event callback
     */
    on(event, callback) {
        this.events.addEventListener(event, callback)
    }

    /**
     * Remove an event listener
     * @param {string} event - Event name
     * @param {Function} callback - Event callback
     */
    off(event, callback) {
        this.events.removeEventListener(event, callback)
    }

    /**
     * Dispatch an event
     * @private
     * @param {string} event - Event name
     * @param {Object} detail - Event details
     */
    _dispatchEvent(event, detail) {
        this.events.dispatchEvent(new CustomEvent(event, { detail }))
    }

    /**
     * Log a message if debug is enabled
     * @private
     * @param {...any} args - Arguments to log
     */
    log(...args) {
        if (this.options.debug) {
            console.log('[Decentifai]', ...args)
        }
    }

    /**
     * Log a message if debug is enabled
     * @private
     * @param {...any} args - Arguments to log
     */
    warn(...args) {
        if (this.options.debug) {
            console.warn('[Decentifai]', ...args)
        }
    }

    /**
     * Disconnect from the federated learning network
     */
    disconnect() {
        this.log('Disconnecting from federated learning network')

        // Clean up WebRTC connections
        if (this.provider) {
            this.provider.destroy()
        }

        // Clean up Y.js document
        // this.ydoc.destroy()

        this._dispatchEvent('disconnected', {})
    }
    /**
     * Peer added event.
     * @event Decentifai#peersAdded
     * @type {object}
     * @property {string[]} peers - Array of connected peer IDs
     */

    /**
     * Peer information changed event.
     * @event Decentifai#peersChanged
     * @type {object}
     * @property {string[]} peers - Array of connected peer IDs
     */

    /**
     * Peer removed event.
     * @event Decentifai#peersRemoved
     * @type {object}
     * @property {string[]} peers - Array of remaining connected peer IDs
     */

    /**
     * Parameters received event.
     * @event Decentifai#parametersReceived
     * @type {object}
     * @property {object} parameters - Object containing received parameters
     * @property {string} source - ID of the peer that sent the parameters
     */

    /**
     * Parameters shared event.
     * @event Decentifai#parametersShared
     * @type {object}
     * @property {object} parameters - Object containing the shared parameters
     */

    /**
     * Parameters applied event.
     * @event Decentifai#parametersApplied
     * @type {object}
     * @property {object} parameters - Object containing the parameters that were applied
     */

    /**
     * Round changed event.
     * @event Decentifai#roundChanged
     * @type {object}
     * @property {number} round - The new training round number
     */

    /**
     * Round proposed event.
     * @event Decentifai#roundProposed
     * @type {object}
     * @property {number} round - The proposed training round number
     */

    /**
     * Round quorum reached event.
     * @event Decentifai#roundQuorumReached
     * @type {object}
     * @property {number} round - The training round number that reached quorum
     */

    /**
     * Round started event.
     * @event Decentifai#roundStarted
     * @type {object}
     * @property {number} round - The training round number that started
     */

    /**
     * Local training completed event.
     * @event Decentifai#localTrainingCompleted
     * @type {object}
     * @property {number} round - The current training round number
     * @property {boolean} isConverged - Boolean indicating if the model has converged
     * @property {object} modelInfo - Object containing training metrics from the model
     */

    /**
     * Round finalized event.
     * @event Decentifai#roundFinalized
     * @type {object}
     * @property {number} round - The training round number that was finalized
     * @property {number} participants - Number of peers that participated in the round
     * @property {object} parameters - The aggregated parameters from all peers
     */

    /**
     * Auto-training started event.
     * @event Decentifai#autoTrainingStarted
     * @type {object}
     */

    /**
     * Auto-training round completed event.
     * @event Decentifai#autoTrainingRoundCompleted
     * @type {object}
     * @property {number} round - The training round number that completed
     * @property {boolean} isConverged - Boolean indicating if the model has converged
     */

    /**
     * Auto-training stopped event.
     * @event Decentifai#autoTrainingStopped
     * @type {object}
     * @property {string} reason - Reason why auto-training stopped (e.g., 'maxRoundsReached')
     */

    /**
     * Auto-training paused event.
     * @event Decentifai#autoTrainingPaused
     * @type {object}
     * @property {string} reason - Reason why auto-training was paused (e.g., 'insufficientPeers')
     */

    /**
     * Auto-training error event.
     * @event Decentifai#autoTrainingError
     * @type {object}
     * @property {string} error - Error message describing what went wrong
     */

    /**
     * Model converged event.
     * @event Decentifai#modelConverged
     * @type {object}
     * @property {number} round - The training round at which convergence was detected
     * @property {object} convergenceMetrics - Object containing convergence metrics
     */

    /**
     * Disconnected event.
     * @event Decentifai#disconnected
     * @type {object}
     */
}