// import { Doc, WebrtcProvider, awarenessProtocol } from "https://prafulb.github.io/bundledYjs/dist/yjs-bundle.esm.js"
import { Doc } from "https://esm.sh/yjs"

const DEFAULTS = {
    // connectionType: "webrtc",
    APP_ID: "Decentifai",
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
     * This class also allows for creation and access to custom shared Y.Map and Y.Array structures
     * for generic data sharing (e.g., embeddings) using `getCustomSharedMap` and `getCustomSharedArray`.
     * Users are responsible for attaching observers to these custom structures.
     * * @param {Object} options - Configuration options for Decentifai
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
     * @param {Number} options.federationOptions.convergenceThresholds.parameterDistance - RMS distance of model parameters across stability window before convergence (check disabled for now)
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
            appId: DEFAULTS.APP_ID,
            roomId: 'Decentifai-test',
            connectionType: 'webrtc',
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
        if (!this.model.extractLocalParametersFunc || typeof (this.model.extractLocalParametersFunc) !== 'function') {
            if (!this.onDeviceTraining) {
                throw new Error('Function to Extract Local Parameters must be specified for generic models.')
            }
            this._getDefaultExtractFunction(this.options.backend).then(func => {
                this.model.extractLocalParametersFunc = func
            })
        }

        if (!this.model.updateLocalParametersFunc || typeof (this.model.updateLocalParametersFunc) !== 'function') {
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
        this._setupP2P().then(() => {
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
     * @returns {Promise<Function>} - Parameter extraction function
     */
    async _getDefaultExtractFunction(backend) {
        const { extractLocalParameters } = await import(`${DEFAULTS.appBasePath}/utils/extractors.js`)
        return extractLocalParameters(backend)
    }

    /**
     * Get default parameter update function based on backend
     * @private
     * @param {string} backend - ML backend type ('tfjs' or 'onnx')
     * @returns {Promise<Function>} - Parameter update function
     */
    async _getDefaultUpdateFunction(backend) {
        const { updateLocalParameters } = await import(`${DEFAULTS.appBasePath}/utils/updators.js`)
        return updateLocalParameters(backend)
    }

    /**
     * Get default parameter aggregation function (Federated Averaging)
     * @private
     * @param {string} aggregationMethod - The name of the aggregation method.
     * @returns {Promise<Function>} - Parameter aggregation function
     */
    async _getAggregator(aggregationMethod) {
        const aggregators = await import(`${DEFAULTS.appBasePath}/utils/aggregators.js`)
        return aggregators[aggregationMethod]
    }

    /**
     * Get default model training function based on backend.
     * Placeholder: Implement this if you have default training functions.
     * @private
     * @param {string} backend - ML backend type ('tfjs' or 'onnx')
     * @returns {Promise<Function>} - Model training function
     */
    async _getDefaultModelTrainingFunction(backend) {
        // Example:
        // if (backend === 'tfjs' || backend === 'onnx') {
        //     const { trainModel } = await import(`${DEFAULTS.appBasePath}/utils/trainers.js`)
        //     return trainModel(backend)
        // }
        this.log(`_getDefaultModelTrainingFunction called for ${backend}, but no default trainer is set up.`)
        return async () => { throw new Error(`Default training function for ${backend} not implemented.`) }
    }


    /**
     * Initialize the Y.js document
     * @private
     */
    _initYDoc() {
        this.ydoc = new Doc()

        // Create shared data structures for federated learning
        this.parameters = this.ydoc.getMap('parameters')
        this.metadata = this.ydoc.getMap('metadata')
        this.roundInfo = this.ydoc.getMap('roundInfo')

        this.metadata.set(this.ydoc.clientID.toString(), { peerId: this.ydoc.clientID, ...this.options.metadata })

        this.log('Y.js document initialized')
    }

    /**
     * Set up WebRTC connections using y-webrtc
     * @private
     */

    async _setupP2P() {
        if (this.options.connectionType === 'webrtc') {
            const { WebrtcProvider } = await import("https://prafulb.github.io/bundledYjs/y-webrtc.js")

            let iceServers = this.options.iceServers
            if (iceServers.length === 0) {
                for (const iceServer of DEFAULTS.iceServers) {
                    let processedIceServer = { ...iceServer }
                    if (processedIceServer.credsRequired) {
                        try {
                            const response = await fetch(processedIceServer.getCredsFrom)
                            if (!response.ok) {
                                throw new Error(`Failed to fetch TURN creds: ${response.statusText}`)
                            }
                            const { username, credential } = await response.json()
                            processedIceServer.username = username
                            processedIceServer.credential = credential
                            delete processedIceServer.getCredsFrom
                            delete processedIceServer.credsRequired
                        } catch (error) {
                            this.warn(`Could not fetch credentials for TURN server ${processedIceServer.urls}. Error: ${error.message}`)
                            continue
                        }
                    }
                    iceServers.push(processedIceServer)
                }
            }
            const webrtcProviderOptions = {
                signaling: this.options.signaling,
                password: this.options.federationOptions?.password,
                // awareness: new awarenessProtocol.Awareness(this.ydoc),
                maxConns: this.options.federationOptions.maxPeers || DEFAULTS.maxConns,
                filterBcConns: true,
                peerOpts: {
                    config: {
                        iceServers
                    }
                }
            }
            if(this.options.federationOptions.password){
                webrtcProviderOptions.password = this.options.federationOptions.password
            }
    
            this.provider = new WebrtcProvider(this.options.roomId, this.ydoc, webrtcProviderOptions)

        } else {
            const { TrysteroProvider } = await import("https://esm.sh/@winstonfassett/y-webrtc-trystero")
            const { joinRoom } = await import("https://esm.sh/trystero/nostr")

            const trysteroRoom = joinRoom({ 
                appId: this.options.appId || DEFAULTS.APP_ID,
                password: this.options.federationOptions.password
            }, this.options.roomId)
            this.provider = new TrysteroProvider(this.options.roomId, this.ydoc, {
                trysteroRoom,
                maxConns: this.options.federationOptions.maxPeers || DEFAULTS.maxConns
            })
        }

        this.awareness = this.provider.awareness
        // console.log(this.awareness)

        // Set up awareness
        this.awareness.setLocalState({
            clientID: this.ydoc.clientID,
            name: this.metadata?.name,
            lastSeen: Date.now(),
            online: true,
            training: false,
            round: this.trainingRound,
            roundStatus: 'idle', // Initial round status
            metadata: { name: `Peer-${this.ydoc.clientID.toString().slice(-4)}`, ...this.options.metadata } // Add a default name
        })

        // Handle peer connections
        this.awareness.on('change', (changes) => {
            this._handlePeerUpdate(changes)
        })

        // this.provider.on('peers', (event) => {
        //     this._handlePeerUpdate(event)
        // })


        // Handle parameter updates
        this.parameters.observe(event => {
            this._handleParameterUpdate(event)
        })

        // Handle round info updates
        this.roundInfo.observe(event => {
            this._handleRoundUpdate(event)
        })

        this.log('WebRTC provider initialized with room:', this.options.roomId)
        this.log(`Self ID [${this.ydoc.clientID}] (${this.awareness.getLocalState()?.metadata?.name}) Available to connect with peers`)
    }

    /**
     * Get a Y.Map for custom data sharing (e.g., embeddings).
     * The user is responsible for setting up observers on the returned map.
     * @param {string} name - The unique name for this shared map. This name should not conflict with internal names: 'parameters', 'metadata', 'roundInfo'.
     * @returns {Object | null} The Y.Map instance, or null if the name is reserved.
     */
    getCustomSharedMap(name) {
        if (['parameters', 'metadata', 'roundInfo'].includes(name)) {
            this.warn(`The name "${name}" is reserved for internal Decentifai use. Please choose a different name for your custom shared map.`)
            return null
        }
        this.log(`Accessing/Creating custom shared Y.Map: ${name}`)
        return this.ydoc.getMap(name)
    }

    /**
     * Get a Y.Array for custom data sharing (e.g., a list of embeddings).
     * The user is responsible for setting up observers on the returned array.
     * @param {string} name - The unique name for this shared array. This name should not conflict with internal names: 'parameters', 'metadata', 'roundInfo'.
     * @returns {Object | null} The Y.Array instance, or null if the name is reserved.
     */
    getCustomSharedArray(name) {
        if (['parameters', 'metadata', 'roundInfo'].includes(name)) {
            this.warn(`The name "${name}" is reserved for internal Decentifai use. Please choose a different name for your custom shared array.`)
            return null
        }
        this.log(`Accessing/Creating custom shared Y.Array: ${name}`)
        return this.ydoc.getArray(name)
    }

    /**
     * Get WebRTC peer ID of current instance as seen by others
     * @returns {number} The clientID of this Y.Doc instance.
     */
    getSelfPeerId() {
        return this.ydoc.clientID
    }

    /**
     * Handle peer connection updates based on awareness changes.
     * @private
     * @param {Object} event - Peer update event (from y-webrtc 'peers' event which has {added, removed, webrtcPeers, bcPeers})
     */
    _handlePeerUpdate(event) {
        const { added, updated, removed } = event
        const allStates = this.awareness.getStates()
        const currentPeerIds = new Set()

        allStates.forEach(state => {
            if (state.online && state.clientID !== this.ydoc.clientID) {
                currentPeerIds.add(state.clientID)
                const peerExists = !!this.peers[state.clientID]
                this.peers[state.clientID] = {
                    clientID: state.clientID,
                    connected: true,
                    lastSeen: state.lastSeen || Date.now(),
                    metadata: state.metadata,
                    awarenessState: { ...state }
                }
                if (!peerExists) {
                    this.log(`New peer ${state.metadata?.name || state.clientID} connected.`)
                    this._dispatchEvent('peersAdded', { peerId: state.clientID, name: state.metadata?.name, peers: Object.keys(this.peers) })
                } else {
                    // this.log(`Peer ${state.metadata?.name || state.clientID} updated.`)
                    this._dispatchEvent('peersChanged', { peerId: state.clientID, name: state.metadata?.name, peers: Object.keys(this.peers) })
                }
            }
        })

        // Handle removals for peers no longer in awareness or marked offline
        Object.keys(this.peers).forEach(existingPeerId => {
            if (!currentPeerIds.has(parseInt(existingPeerId))) { // clientID is a number
                const peer = this.peers[existingPeerId]
                this.log(`Peer ${peer.metadata?.name || existingPeerId} disconnected or went offline.`)
                delete this.peers[existingPeerId]
                this._dispatchEvent('peersRemoved', { peerId: parseInt(existingPeerId), name: peer.metadata?.name, peers: Object.keys(this.peers) })
            }
        })
    }

    /**
     * Handle parameter updates from peers
     * @private
     * @param {Object} event - Parameter update event
     */
    _handleParameterUpdate(event) {
        if (this.isTraining && event.transaction.local) {
            return
        }

        const updatedParamsByPeer = {}
        let receivedFromOthers = false

        event.changes.keys.forEach((change, key) => {
            const paramData = this.parameters.get(key)

            // Check if the update is from another peer
            if (paramData && paramData.peerId !== this.getSelfPeerId()) {
                if (change.action === 'add' || change.action === 'update') {
                    if (!updatedParamsByPeer[paramData.peerId]) {
                        updatedParamsByPeer[paramData.peerId] = []
                    }
                    updatedParamsByPeer[paramData.peerId].push(paramData)
                    receivedFromOthers = true
                    this.log(`Parameter update received from peer ${paramData.peerId} for round ${paramData.round}`)
                }
            }
        })

        if (receivedFromOthers) {
            this._dispatchEvent('parametersReceived', {
                parametersByPeer: updatedParamsByPeer,
                sourceDescription: event.transaction.local ? 'local-transaction' : `remote-transaction-origin:${event.transaction.origin}`
            })
        }
    }

    /**
     * Handle round info updates
     * @private
     * @param {Object} event - Round update event
     */
    async _handleRoundUpdate(event) {
        // Ensure roundData is always fetched after an event, as it might have changed.
        const roundData = this.roundInfo.get('roundData')

        if (!roundData) {
            this.log('RoundInfo updated but no roundData found.')
            return
        }

        this.log(`Handling round update. Current local round: ${this.trainingRound}. Event roundData:`, JSON.parse(JSON.stringify(roundData)))

        if (event.transaction.local) {
            this.log('Ignoring local roundInfo update in _handleRoundUpdate.')
            return
        }

        // If the incoming round is ahead of the local round
        if (roundData.currentRound > this.trainingRound) {
            this.log(`Incoming round ${roundData.currentRound} is ahead of local round ${this.trainingRound}.`)

            if (this.isTraining) {
                await new Promise(res => this.on('roundFinalized', res, {once: true}))
            }

            if (roundData.status === 'proposed') {
                this.log(`Acknowledging proposal for round ${roundData.currentRound}.`)

                // Forcibly update parameters if this peer is significantly behind (maybe because it missed a full round's aggregation)
                if (this.trainingRound < roundData.currentRound - 1 && roundData.currentRound > 1) {
                    this.log(`Peer is behind by more than one round. Attempting to apply parameters from round ${roundData.currentRound - 1}.`)
                    const previousRoundParameters = this.getParameters(roundData.currentRound - 1)

                    if (previousRoundParameters.length > 0) {
                        const aggregatedParams = this.options.federationOptions.aggregateParametersFunc(previousRoundParameters, this.options.backend)

                        if (aggregatedParams) {
                            this.model.updateLocalParametersFunc(this.model, aggregatedParams)
                                .then(() => this.log(`Forcibly updated model to parameters from round ${roundData.currentRound - 1}.`))
                                .catch(err => this.warn(`Failed to forcibly update model: ${err.message}`))
                        } else {
                            this.warn(`Could not aggregate parameters for catch-up from round ${roundData.currentRound - 1}.`)
                        }
                    } else {
                        this.warn(`No parameters found for catch-up from round ${roundData.currentRound - 1}.`)
                    }

                }

                this.trainingRound = roundData.currentRound

                const localAwarenessState = this.awareness.getLocalState() || {}
                localAwarenessState.round = this.trainingRound
                localAwarenessState.roundStatus = 'acknowledged'
                this.awareness.setLocalState(localAwarenessState)

                this.log(`Updated local round to ${this.trainingRound} and acknowledged.`)
                this._dispatchEvent('roundChanged', { round: this.trainingRound, status: 'acknowledged' })

            } else if (roundData.status === 'training') {
                this.log(`Catching up: round ${roundData.currentRound} is already in training state. Updating local round.`)

                this.trainingRound = roundData.currentRound
                this.startTrainingRound().catch(err => this.warn(`Error starting catch-up training round: ${err.message}`))

                this._dispatchEvent('roundChanged', { round: this.trainingRound, status: 'training' })

            } else if (roundData.status === 'completed') {
                this.log(`Catching up: round ${roundData.currentRound} is already completed. Updating local round.`)

                this.trainingRound = roundData.currentRound
                this.finalizeRound().catch(err => this.warn(`Error during catch-up finalization for round ${this.trainingRound}: ${err.message}`))

                this._dispatchEvent('roundChanged', { round: this.trainingRound, status: 'completed' })
            }
        }

        // If the incoming round is the same as the local round, and the status is changing
        else if (roundData.currentRound === this.trainingRound) {
            const localAwarenessState = this.awareness.getLocalState() || {}

            if (roundData.status === 'training' && localAwarenessState.roundStatus !== 'training') {
                this.log(`Remote peer initiated training for current round ${this.trainingRound}. Starting local training.`)

                this.startTrainingRound().catch(err => this.warn(`Error starting training round prompted by remote: ${err.message}`))
            } else if (roundData.status === 'completed' && localAwarenessState.roundStatus !== 'completed') {
                this.log(`Remote peer finalized round ${this.trainingRound}. Finalizing locally.`)

                this.finalizeRound().catch(err => this.warn(`Error finalizing round prompted by remote: ${err.message}`))
            }

        } else {
            this.log(`Incoming round ${roundData.currentRound} is behind or same as local round ${this.trainingRound} with no actionable status change. Ignoring.`)
        }
    }

    /**
     * Set up auto-training event listeners and initialization
     * @private
     */
    _setupAutoTraining() {
        if (!this.trainingData && this.onDeviceTraining) {
            this.log('Warning: Auto-training enabled but no training data provided and on-device training is expected.')
            // return; // Allow setup even if data isn't immediately available, it might be set later.
        }

        const checkAndStart = () => {
            if (this.autoTrainingEnabled && !this.isTraining && !this.converged) {
                this._checkShouldStartTraining()
            }
        }

        // Listen for peer changes to potentially start training
        this.on('peersAdded', checkAndStart)

        this.on('peersRemoved', () => { // If peers drop below min, auto-training might pause or stop

            if (this.autoTrainingEnabled && this.isTraining) {

                if (Object.keys(this.peers).length + 1 < this.options.federationOptions.minPeers) {
                    this.log('Auto-training paused: not enough peers during a round.')
                    this._dispatchEvent('autoTrainingPaused', { reason: 'insufficientPeers' })
                }

            }
        })

        // Listen for round completion to continue training if needed
        this.on('roundFinalized', () => {
            if (this.autoTrainingEnabled && !this.converged) {
                this._continueTrainingIfNeeded()
            }
        })

        // Listen for model convergence to stop auto-training
        this.on('modelConverged', () => {
            this.converged = true
            this.isTraining = false

            this.log('Auto-training stopped: model converged')
            this._dispatchEvent('autoTrainingStopped', { reason: 'modelConverged' })

            // Update awareness
            const awarenessState = this.awareness.getLocalState() || {}
            awarenessState.training = false
            this.awareness.setLocalState(awarenessState)
        })

        this.log('Auto-training listeners set up.')
        checkAndStart()
    }

    /**
     * Check if auto-training should begin
     * @private
     */
    _checkShouldStartTraining() {
        if (!this.autoTrainingEnabled || this.isTraining || this.converged) {
            this.log(`Auto-training check: enabled=${this.autoTrainingEnabled}, training=${this.isTraining}, converged=${this.converged}. No action.`)
            return
        }

        const numTotalPeers = Object.keys(this.peers).length + 1
        this.log(`Checking peer count for auto-training: ${numTotalPeers} vs minPeers ${this.options.federationOptions.minPeers}`)

        if (numTotalPeers >= this.options.federationOptions.minPeers) {
            this._startAutoTraining().catch(err => {
                this.warn('Error starting auto-training:', err.message)
                this._dispatchEvent('autoTrainingError', { error: `Failed to start auto-training: ${err.message}` })
            })
        } else {
            this.log('Not enough peers to start auto-training.')
        }
    }

    /**
     * Start the automatic training process
     * @private
     */
    async _startAutoTraining() {
        if (this.isTraining || this.converged) {
            this.log('Auto-training start called, but already training or converged.')
            return
        }

        this.log('Attempting to start auto-training process...')

        this.isTraining = true
        this._dispatchEvent('autoTrainingStarted', {})

        try {
            await this._runTrainingRound()
        } catch (error) {
            this.warn('Error during the first auto-training round:', error)
            this.isTraining = false
            this._dispatchEvent('autoTrainingError', { error: `Initial auto-training round failed: ${error.message}` })
        }
    }

    as

    /**
     * Continue training if the model hasn't converged.
     * @private
     */
    async _continueTrainingIfNeeded() {
        if (!this.autoTrainingEnabled || this.isTraining || this.converged) {
            this.log(`Continue training check: autoEnabled=${this.autoTrainingEnabled}, isTraining=${this.isTraining}, converged=${this.converged}. No action.`)
            return
        }

        if (this.trainingRound >= this.options.federationOptions.maxRounds) {
            this.log(`Auto-training stopped: maximum rounds (${this.options.federationOptions.maxRounds}) reached at round ${this.trainingRound}.`)

            this.isTraining = false
            this._dispatchEvent('autoTrainingStopped', { reason: 'maxRoundsReached' })

            // Update awareness
            const awarenessState = this.awareness.getLocalState() || {}
            awarenessState.training = false
            this.awareness.setLocalState(awarenessState)

            return
        }

        // Add a small delay between rounds to allow network synchronization
        this.log(`Waiting ${this.options.federationOptions.waitTime}ms before starting next auto-training round.`)
        await new Promise(resolve => setTimeout(resolve, this.options.federationOptions.waitTime))

        // Check if peers are still connected. Just me being paranoid.
        const numTotalPeers = 1 + Object.keys(this.peers).length
        if (numTotalPeers < this.options.federationOptions.minPeers) {
            this.log(`Auto-training paused: not enough peers (${numTotalPeers}/${this.options.federationOptions.minPeers}).`)

            this.isTraining = false
            this._dispatchEvent('autoTrainingPaused', { reason: 'insufficientPeers' })

            // Update awareness
            const awarenessState = this.awareness.getLocalState() || {}
            awarenessState.training = false
            this.awareness.setLocalState(awarenessState)

            return
        }

        this.log('Proceeding to next auto-training round.')
        this.isTraining = true
        try {
            await this._runTrainingRound()
        } catch (error) {
            this.warn(`Error during auto-training round ${this.trainingRound + 1}:`, error)
            this.isTraining = false
            this._dispatchEvent('autoTrainingError', { error: `Auto-training round failed: ${error.message}` })
        }
    }

    /**
     * Check how many distinct peers shared parameter updates for the current round.
     * @returns {number} Number of unique peers that shared parameters for the current training round.
     */
    getNumPeersWhoSharedParameters() {
        const peerParameters = this.getParameters(this.trainingRound)
        const distinctPeerIds = new Set(peerParameters.map(p => p.peerId))
        return distinctPeerIds.size
    }

    /**
     * Run a single training round in auto-training mode.
     * @private
     */
    async _runTrainingRound() {
        if (!this.trainingData && this.onDeviceTraining) {
            this.warn('Attempted to run training round without training data for on-device model.')

            this.isTraining = false
            this._dispatchEvent('autoTrainingError', { error: 'Missing training data for on-device model.' })

            return
        }

        this.log(`Starting auto-training round ${this.trainingRound + 1} (current round is ${this.trainingRound})`)
        this.isTraining = true

        try {
            // Random wait to reduce simultaneous proposals.
            // TODO high priority: build an election mechanism so that only one peer proposes at any one point.
            await new Promise(res => setTimeout(res, Math.random() * (this.options.federationOptions.waitTime / 2)))

            const currentRoundInfo = this.roundInfo.get('roundData')
            // Only propose if no proposal for this round or next round exists from another peer,
            // or if current round is completed and we are moving to the next. SO CONVOLUTED RIGHT NOW!!!!!!
            let shouldPropose = true

            if (currentRoundInfo) {

                if (currentRoundInfo.currentRound > this.trainingRound && ['proposed', 'training'].includes(currentRoundInfo.status)) {
                    this.log(`Another peer already proposed/started round ${currentRoundInfo.currentRound}. This peer will follow.`)
                    shouldPropose = false
                } else if (currentRoundInfo.currentRound === this.trainingRound + 1 && ['proposed', 'training'].includes(currentRoundInfo.status)) {
                    this.log(`Another peer already proposed/started the target round ${this.trainingRound + 1}. This peer will follow.`)
                    shouldPropose = false
                }

            }

            let proposalQuorumReached = false

            if (shouldPropose) {
                proposalQuorumReached = await this.proposeTrainingRound()
            } else {
                // If another peer proposed, ensure trainingRound is up to date
                if (currentRoundInfo && currentRoundInfo.currentRound > this.trainingRound) {
                    this.trainingRound = currentRoundInfo.currentRound

                    const localAwarenessState = this.awareness.getLocalState() || {}
                    if (localAwarenessState.round !== this.trainingRound || localAwarenessState.roundStatus !== 'acknowledged') {
                        localAwarenessState.round = this.trainingRound
                        localAwarenessState.roundStatus = 'acknowledged'
                        this.awareness.setLocalState(localAwarenessState)
                        this.log(`Following external proposal for round ${this.trainingRound}. Acknowledged.`)
                    }
                }

                await new Promise(res => setTimeout(res, this.options.federationOptions.waitTime * 3)); // Wait for proposal to propagate and be acked
                proposalQuorumReached = this._checkRoundAcknowledgments(this.trainingRound)
            }

            await new Promise(res => setTimeout(res, this.options.federationOptions.waitTime))

            if (proposalQuorumReached || (this.roundInfo.get('roundData')?.currentRound === this.trainingRound && this.roundInfo.get('roundData')?.status === 'training')) {
                this.log(`Quorum reached or training already started for round ${this.trainingRound}. Proceeding to local training.`)
                const modelInfo = await this.startTrainingRound()
                if (!modelInfo && this.onDeviceTraining) {
                    this.warn(`Local training for round ${this.trainingRound} did not return info or may have failed.`)
                }
            } else {
                this.log(`Could not reach quorum for round ${this.trainingRound} or training not started by others. Skipping this round attempt.`)

                this.trainingRound--
                this.isTraining = false

                return
            }

            this.log(`Waiting for peers to share parameters for round ${this.trainingRound}. Timeout: ${10 * this.options.federationOptions.waitTime}ms`)

            let parameterSharingCountdown = 10

            const enoughParametersShared = await new Promise(resolve => {
                const checkIfEnoughParametersShared = setInterval(() => {
                    parameterSharingCountdown--
                    const numPeersWhoShared = this.getNumPeersWhoSharedParameters()
                    this.log(`Parameter check: ${numPeersWhoShared}/${this.options.federationOptions.minPeers} peers shared parameters for round ${this.trainingRound}. Countdown: ${parameterSharingCountdown}`)
                    if (numPeersWhoShared >= this.options.federationOptions.minPeers) {
                        clearInterval(checkIfEnoughParametersShared)
                        resolve(true)
                    } else if (parameterSharingCountdown === 0) {
                        clearInterval(checkIfEnoughParametersShared)
                        resolve(false)
                    }
                }, this.options.federationOptions.waitTime)
            })

            if (enoughParametersShared) {
                this.log(`Sufficient parameters received for round ${this.trainingRound}. Proceeding to finalize.`)

                const finalized = await this.finalizeRound()

                if (finalized) {
                    this._dispatchEvent('autoTrainingRoundCompleted', {
                        round: this.trainingRound,
                        isConverged: this.converged
                    })
                } else {
                    this.warn(`Round ${this.trainingRound} finalization failed despite having enough parameters initially.`)
                }

            } else {
                this.log(`Timeout waiting for parameters from peers for round ${this.trainingRound}. Aggregation quorum not met. Round cannot be finalized.`)
            }

        } catch (error) {
            this.warn(`Error in auto-training round execution (round ${this.trainingRound}):`, error)
            this._dispatchEvent('autoTrainingError', { round: this.trainingRound, error: error.message })
        } finally {
            this.isTraining = false

            const awarenessState = this.awareness.getLocalState() || {}
            awarenessState.training = false
            // awarenessState.roundStatus = this.converged ? 'converged' : (this.roundInfo.get('roundData')?.status || 'idle_after_round')
            this.awareness.setLocalState(awarenessState)
            this.log(`Finished auto-training round attempt ${this.trainingRound}.`)
        }
    }

    /**
     * Enable or disable automatic training
     * @param {boolean} enable - Whether to enable auto-training
     * @param {Object|Array|Function} [trainingData=null] - Training data to use (optional)
     * @param {Object} [trainingOptions=null] - Options for training (optional)
     * @returns {boolean} The new state of autoTrainingEnabled.
     */
    setAutoTraining(enable, trainingData = null, trainingOptions = null) {
        this.autoTrainingEnabled = !!enable

        if (trainingData !== null) {
            this.trainingData = trainingData
        }

        if (trainingOptions !== null) {
            this.options.trainingOptions = { ...this.options.trainingOptions, ...trainingOptions }
        }

        this.log(`Auto-training ${this.autoTrainingEnabled ? 'enabled' : 'disabled'}`)

        if (this.autoTrainingEnabled) {

            if (!this.events.listeners?.['peersAdded']?.some(l => l.toString().includes('_checkShouldStartTraining'))) {
                this._setupAutoTraining()
            }
            this._checkShouldStartTraining()

        } else {
            this.isTraining = false

            const awarenessState = this.awareness.getLocalState() || {}
            awarenessState.training = false
            this.awareness.setLocalState(awarenessState)
        }

        return this.autoTrainingEnabled
    }

    /**
     * Check if conditions are met for manual training initiation.
     * @returns {boolean} True if manual training can be initiated.
     */
    canTrainManually() {
        const numTotalPeers = Object.keys(this.peers).length + 1
        if (!this.autoTrainingEnabled && numTotalPeers >= this.options.federationOptions.minPeers && !this.isTraining) {
            this.log('Manual training can be initiated: conditions met.')
            return true
        }
        this.log(`Manual training conditions not met: autoTraining=${this.autoTrainingEnabled}, peerCount=${numTotalPeers}/${this.options.federationOptions.minPeers}, isTraining=${this.isTraining}`)
        return false
    }

    /**
     * Train the local model on local data and share parameters with peers afterwards.
     * @param {Array|Object|null} data - Training data. If null and onDeviceTraining is true, an error will be thrown.
     * @param {Object} [options={}] - Training options to pass to the model's `train` function
     * @returns {Promise<any>} - Resolves with model training output when training is complete
     */
    async trainLocal(data, options = {}) {
        if (!data && this.onDeviceTraining) {
            throw new Error('Training data must be provided for on-device training models.')
        }
        if (typeof this.model.train !== 'function') {
            throw new Error('Model must have a `train` method.')
        }

        this.log(`Starting local training for round ${this.trainingRound}.`)
        this.isTraining = true

        // Update awareness state
        const localAwarenessState = this.awareness.getLocalState() || {}
        localAwarenessState.training = true
        localAwarenessState.round = this.trainingRound
        localAwarenessState.roundStatus = 'training_local'
        this.awareness.setLocalState(localAwarenessState)

        try {
            const trainingArgs = this.onDeviceTraining ? { data, options: { ...this.options.trainingOptions, ...options } } : { options: { ...this.options.trainingOptions, ...options } }
            const info = await this.model.train(trainingArgs)

            this.log(`Local training completed for round ${this.trainingRound}.`)
            this._dispatchEvent('localTrainingCompleted', {
                round: this.trainingRound,
                isConverged: this.converged,
                modelInfo: info
            })

            const params = await this.model.extractLocalParametersFunc(this.model)
            this.shareParameters(params)

            return info

        } catch (error) {
            this.warn(`Local training error in round ${this.trainingRound}:`, error)
            this._dispatchEvent('autoTrainingError', { round: this.trainingRound, error: `Local training failed: ${error.message}` })
            // throw error
        } finally {
            const finalAwarenessState = this.awareness.getLocalState() || {}
            finalAwarenessState.roundStatus = 'training_done_sharing'
            this.awareness.setLocalState(finalAwarenessState)
        }
    }

    /**
     * Share local model parameters with peers for the current training round.
     * @param {Object} [parameters] - The model parameters to share. If not provided, they are extracted.
     */
    async shareParameters(parameters) {
        let paramsToShare = parameters
        if (!paramsToShare) {
            if (typeof this.model.extractLocalParametersFunc !== 'function') {
                this.warn('Cannot share parameters: extractLocalParametersFunc is not defined.')
                return
            }
            paramsToShare = await this.model.extractLocalParametersFunc(this.model)
        }

        if (!paramsToShare) {
            this.warn('No parameters extracted to share.')
            return
        }

        this.log(`Sharing parameters for round ${this.trainingRound} from peer ${this.ydoc.clientID}.`)

        // Add metadata to parameters
        const paramUpdate = {
            peerId: this.ydoc.clientID,
            timestamp: Date.now(),
            round: this.trainingRound, // Ensure this is the correct current round
            parameters: paramsToShare
        }

        this.parameters.set(`params_${this.ydoc.clientID}_round_${this.trainingRound}`, paramUpdate)

        this._dispatchEvent('parametersShared', { parameters: paramUpdate })
    }

    /**
     * Propose a new training round to all peers.
     * @returns {Promise<boolean>} - True if quorum was reached for the proposal in the specified wait time.
     */
    async proposeTrainingRound() {
        const nextRound = this.trainingRound + 1
        this.log(`Proposing training round ${nextRound}.`)

        // Check if this round has already been effectively proposed or started by another peer.
        const currentRoundData = this.roundInfo.get('roundData')
        if (currentRoundData && currentRoundData.currentRound === nextRound && ['proposed', 'training'].includes(currentRoundData.status)) {

            this.log(`Round ${nextRound} already proposed/in training by ${currentRoundData.initiator}. This peer will acknowledge and follow.`)

            this.trainingRound = nextRound

            const localAwarenessState = this.awareness.getLocalState() || {}
            localAwarenessState.round = this.trainingRound
            localAwarenessState.roundStatus = 'acknowledged'
            this.awareness.setLocalState(localAwarenessState)

            this._dispatchEvent('roundChanged', { round: this.trainingRound, status: 'acknowledged' })
            return this._checkRoundAcknowledgments(this.trainingRound); // Check if already acknowledged by enough peers
        }

        this.trainingRound = nextRound

        this.roundInfo.set('roundData', {
            currentRound: this.trainingRound,
            status: 'proposed',
            initiator: this.ydoc.clientID,
            proposeTime: Date.now(),
            acknowledgedBy: { [this.ydoc.clientID]: Date.now() }
        })

        const localAwarenessState = this.awareness.getLocalState() || {}
        localAwarenessState.round = this.trainingRound
        localAwarenessState.roundStatus = 'acknowledged'
        this.awareness.setLocalState(localAwarenessState)

        this._dispatchEvent('roundProposed', { round: this.trainingRound, initiator: this.ydoc.clientID })

        // Wait for acknowledgments
        let quorumCheckCountdown = 10
        const quorumReached = await new Promise((resolve) => {
            const checkForQuorumInterval = setInterval(() => {
                quorumCheckCountdown--
                if (this._checkRoundAcknowledgments(this.trainingRound)) {
                    clearInterval(checkForQuorumInterval)
                    resolve(true)
                } else if (quorumCheckCountdown === 0) {
                    clearInterval(checkForQuorumInterval)
                    this.log(`Timeout waiting for acknowledgments for round ${this.trainingRound}.`)
                    resolve(false)
                }
            }, this.options.federationOptions.waitTime)
        })

        if (quorumReached) {
            this.log(`Quorum reached for round ${this.trainingRound} proposal.`)
            this._dispatchEvent('roundQuorumReached', { round: this.trainingRound })
        } else {
            this.log(`Quorum NOT reached for round ${this.trainingRound} proposal.`)

            this.trainingRound--

            this.roundInfo.delete('roundData')

            const localAwarenessState = this.awareness.getLocalState() || {}
            localAwarenessState.round = this.trainingRound
            localAwarenessState.roundStatus = ''
            this.awareness.setLocalState(localAwarenessState)

        }
        return quorumReached
    }

    /**
     * Check if enough peers acknowledged the specified round.
     * Relies on peers setting their awareness state.
     * @param {number} targetRound - The round number to check acknowledgments for.
     * @returns {boolean} True if quorum of acknowledgments is met.
     */
    _checkRoundAcknowledgments(targetRound) {
        let acknowledgedPeersCount = 0
        const allAwarenessStates = this.awareness.getStates()

        allAwarenessStates.forEach((state, clientID) => {

            if (state.online && state.round === targetRound && state.roundStatus === 'acknowledged') {
                if (clientID === this.ydoc.clientID || this.peers[clientID]?.connected) {
                    acknowledgedPeersCount++
                }
            }

        })

        const minRequired = this.options.federationOptions.minPeers
        if (acknowledgedPeersCount >= minRequired) {
            this.log(`Round ${targetRound} acknowledged by ${acknowledgedPeersCount}/${minRequired} peers (quorum met).`)
            return true
        } else {
            this.log(`Waiting for round ${targetRound} acknowledgments: ${acknowledgedPeersCount}/${minRequired} peers. (Quorum not met)`)
            return false
        }
    }

    /**
     * Start a training round. Should only be called after a round is successfully proposed and acknowledged (quorum met).
     * @returns {Promise<any|null>} Model info from local training, or null if training couldn't start.
     */
    async startTrainingRound() {
        if (this.isTraining && this.roundInfo.get('roundData')?.currentRound === this.trainingRound && this.roundInfo.get('roundData')?.status === 'training') {
            this.log(`Training round ${this.trainingRound} already in progress locally.`)
            return null
        }

        const currentRoundData = this.roundInfo.get('roundData')
        if (!currentRoundData || currentRoundData.currentRound !== this.trainingRound || currentRoundData.status === 'completed') {
            this.warn(`Cannot start training round ${this.trainingRound}. Current round data:`, currentRoundData)
            return null
        }

        this.log(`Attempting to start training for round ${this.trainingRound}.`)

        if (currentRoundData.initiator === this.ydoc.clientID || !currentRoundData.initiator) { // Allow starting if no initiator or self is initiator
            this.roundInfo.set('roundData', {
                ...currentRoundData,
                currentRound: this.trainingRound,
                status: 'training',
                startTime: Date.now()
            })
            this.log(`Round ${this.trainingRound} status set to 'training' by this peer.`)
        } else {
            this.log(`This peer (${this.ydoc.clientID}) is not the initiator (${currentRoundData.initiator}) of round ${this.trainingRound}. Waiting for initiator to set status to 'training'.`)
        }

        this._dispatchEvent('roundStarted', { round: this.trainingRound })
        return await this._startLocalTrainingRound()
    }

    /**
     * Perform the local training for the current round.
     * @private
     * @returns {Promise<any>} Model info from local training.
     */
    async _startLocalTrainingRound() {
        this.log(`Executing local training for round ${this.trainingRound}.`)

        let dataToTrainWith

        if (typeof this.trainingData === 'function') {
            dataToTrainWith = await this.trainingData(this.trainingRound); // Pass current round
        } else {
            dataToTrainWith = this.trainingData
        }

        if (!dataToTrainWith && this.onDeviceTraining) {
            this.warn(`No training data available for on-device model in round ${this.trainingRound}.`)
            this._dispatchEvent('autoTrainingError', { round: this.trainingRound, error: 'Missing training data for local round.' })
            return null
        }

        return await this.trainLocal(dataToTrainWith, this.options.trainingOptions)
    }

    _calculateDistance(params1, params2) {
        // console.log(params1, params2)
        if (!params1 || !params2) return Infinity

        const p1Keys = Object.keys(params1)
        const p2Keys = Object.keys(params2)

        if (p1Keys.length === 0 || p1Keys.length !== p2Keys.length) {
            // This might happen if using TF.js default extract/update which wraps in an array
            if (Array.isArray(params1) && params1.length > 0) params1 = params1[0]
            if (Array.isArray(params2) && params2.length > 0) params2 = params2[0]
            if (!params1 || !params2) return Infinity
        }

        const keys = Object.keys(params1)
        if (keys.length === 0) return 0

        let totalSquaredDifference = 0
        let count = 0

        for (const key of keys) {
            const val1 = params1[key]
            const val2 = params2[key]

            if (val1 === undefined || val2 === undefined) {
                this.warn(`Undefined parameter for key ${key} during distance calculation.`)
                continue; // Skip if one is undefined
            }

            let v1 = val1.values || val1
            let v2 = val2.values || val2

            if (typeof v1 === 'number' && typeof v2 === 'number') {
                totalSquaredDifference += (v1 - v2) ** 2
                count++
            } else if (Array.isArray(v1) && Array.isArray(v2) && v1.length === v2.length) {
                for (let i = 0; i < v1.length; i++) {
                    if (typeof v1[i] === 'number' && typeof v2[i] === 'number') {
                        totalSquaredDifference += (v1[i] - v2[i]) ** 2
                        count++
                    } else {
                        this.warn(`Non-numeric value in array for key ${key} at index ${i}`)
                    }
                }
            } else if (v1 && typeof v1.length === 'number' && v2 && typeof v2.length === 'number' && v1.length === v2.length) { // TypedArrays (e.g. Float32Array)
                for (let i = 0; i < v1.length; i++) {
                    totalSquaredDifference += (v1[i] - v2[i]) ** 2
                    count++
                }
            }
            else {
                this.warn(`Skipping incompatible parameter type for key ${key} in distance calculation. val1:`, val1, "val2:", val2)
            }
        }
        if (count === 0) return Infinity
        return Math.sqrt(totalSquaredDifference / count)
    }

    _checkConvergence() {
        const { parameterDistance, modelLoss, trainingAccuracy } = this.convergenceMetrics
        const stabilityWindow = this.convergenceThresholds.stabilityWindow

        if (this.convergenceHistory.length < stabilityWindow) {
            this.log(`Convergence check: Not enough history (${this.convergenceHistory.length}/${stabilityWindow}).`)
            return false
        }

        // Check parameter distance stability
        const recentDistances = parameterDistance.slice(-stabilityWindow)
        const avgRecentDistance = recentDistances.reduce((sum, d) => sum + d, 0) / stabilityWindow
        const distanceConverged = avgRecentDistance < this.convergenceThresholds.parameterDistance
        // this.log(`Convergence - Parameter Distance: avg recent = ${avgRecentDistance.toFixed(5)}, threshold = ${this.convergenceThresholds.parameterDistance}, converged = ${distanceConverged}`)

        let lossConverged = true; // Default to true if not tracking loss
        if (this.model.getLoss && modelLoss.length >= stabilityWindow) {
            const recentLosses = modelLoss.slice(-stabilityWindow)

            const lossDeltas = []
            for (let i = 1; i < recentLosses.length; i++) {
                lossDeltas.push(Math.abs(recentLosses[i] - recentLosses[i - 1]))
            }

            const avgLossDelta = lossDeltas.reduce((sum, d) => sum + d, 0) / (lossDeltas.length || 1)
            lossConverged = avgLossDelta < this.convergenceThresholds.lossDelta

            this.log(`Convergence - Loss Delta: avg recent delta = ${avgLossDelta.toFixed(5)}, threshold = ${this.convergenceThresholds.lossDelta}, converged = ${lossConverged}`)
        }

        let accuracyConverged = true
        if (this.model.getAccuracy && trainingAccuracy.length >= stabilityWindow) {
            const recentAccuracies = trainingAccuracy.slice(-stabilityWindow)
            const accDeltas = []

            for (let i = 1; i < recentAccuracies.length; i++) {
                accDeltas.push(Math.abs(recentAccuracies[i] - recentAccuracies[i - 1]))
            }

            const avgAccDelta = accDeltas.reduce((sum, d) => sum + d, 0) / (accDeltas.length || 1)
            accuracyConverged = avgAccDelta < this.convergenceThresholds.accuracyDelta

            this.log(`Convergence - Accuracy Delta: avg recent delta = ${avgAccDelta.toFixed(5)}, threshold = ${this.convergenceThresholds.accuracyDelta}, converged = ${accuracyConverged}`)
        }

        // const overallConverged = distanceConverged && lossConverged && accuracyConverged
        const overallConverged = lossConverged && accuracyConverged; // Don't check distance for now, need to think of serialization format for model parameters that works for all types of models.
        if (overallConverged) {
            this.log(`Overall convergence met at round ${this.trainingRound}.`)
        }

        return overallConverged
    }

    async _trackConvergenceMetrics() {
        if (typeof this.model.extractLocalParametersFunc !== 'function') {
            this.warn('Cannot track convergence: extractLocalParametersFunc is missing.')
            return
        }

        const currentParams = await this.model.extractLocalParametersFunc(this.model)
        
        if (this.convergenceHistory.length > 0) {
            const previousHistoryEntry = this.convergenceHistory[this.convergenceHistory.length - 1]
            const previousParams = previousHistoryEntry.modelParameters
            const paramDist = this._calculateDistance(currentParams, previousParams)

            this.convergenceMetrics.parameterDistance.push(paramDist)
            this.log(`Convergence metric: Parameter distance to previous round = ${paramDist.toFixed(5)}`)

        } else {
            this.convergenceMetrics.parameterDistance.push(Infinity)
        }

        if (typeof this.model.getLoss === 'function') {
            let currentLoss = await this.model.getLoss()
            if (Array.isArray(currentLoss)) {
                currentLoss = currentLoss[0]
            }
            
            if (typeof currentLoss === 'number') {
                this.convergenceMetrics.modelLoss.push(currentLoss)
                this.log(`Convergence metric: Current model loss = ${currentLoss.toFixed(5)}`)
            } else {
                this.warn("Model getLoss() did not return a number.")
            }

        }

        if (typeof this.model.getAccuracy === 'function') {
            let currentAccuracy = await this.model.getAccuracy()
            if (Array.isArray(currentAccuracy)) {
                currentAccuracy = currentAccuracy[0]
            }

            if (typeof currentAccuracy === 'number') {
                this.convergenceMetrics.trainingAccuracy.push(currentAccuracy)
                this.log(`Convergence metric: Current model accuracy = ${currentAccuracy.toFixed(5)}`)
            } else {
                this.warn("Model getAccuracy() did not return a number.")
            }
        }

        this.convergenceHistory.push({
            modelParameters: currentParams,
            modelLoss: this.convergenceMetrics.modelLoss.slice(-1)[0],
            trainingAccuracy: this.convergenceMetrics.trainingAccuracy.slice(-1)[0],
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
     * Get parameter sets from all peers that contributed to a given round.
     * @param {number} round - The training round number.
     * @returns {Array<Object>} - Array of parameter objects {peerId, timestamp, round, parameters} from peers for the requested round.
     */
    getParameters(round) {
        const peerParametersForRound = []
        this.parameters.forEach((paramUpdate, key) => {
            if (paramUpdate && paramUpdate.round === round) {
                peerParametersForRound.push(paramUpdate)
            }
        })
        this.log(`${peerParametersForRound.length} parameter sets for round ${round}.`)

        return peerParametersForRound
    }

    /**
     * Finalize a training round by: 1. aggregating parameters, 2. updating local model, and 3. tracking convergence.
     * @returns {Promise<boolean>} - True if the round was successfully finalized, false otherwise.
     */
    async finalizeRound() {
        this.log(`Attempting to finalize training round ${this.trainingRound}.`)

        const peerParameters = this.getParameters(this.trainingRound)
        const numContributingPeers = new Set(peerParameters.map(p => p.peerId)).size

        if (numContributingPeers < this.options.federationOptions.minPeers) {
            this.log(`Cannot finalize round ${this.trainingRound}: Insufficient parameters received (${numContributingPeers}/${this.options.federationOptions.minPeers} unique peers).`)
            return false
        }

        this.log(`Aggregating parameters from ${numContributingPeers} unique peers for round ${this.trainingRound}.`)

        let aggregatedParams
        try {
            aggregatedParams = await this.options.federationOptions.aggregateParametersFunc(peerParameters, this.options.backend, this.model)
        } catch (error) {
            this.warn(`Error during parameter aggregation for round ${this.trainingRound}:`, error)
            return false
        }

        if (!aggregatedParams) {
            this.warn(`Parameter aggregation for round ${this.trainingRound} resulted in null/undefined parameters.`)
            return false
        }

        try {
            await this.model.updateLocalParametersFunc(this.model, aggregatedParams)
            this.log(`Local model updated with aggregated parameters for round ${this.trainingRound}.`)
        } catch (error) {
            this.warn(`Error updating local model with aggregated parameters for round ${this.trainingRound}:`, error)
            return false
        }

        await this._trackConvergenceMetrics()

        if (this._checkConvergence()) {
            this.converged = true
            this.log(`Model has converged at round ${this.trainingRound}.`)
            this._dispatchEvent('modelConverged', {
                round: this.trainingRound,
                convergenceMetrics: this.getConvergenceVisualization()
            })
        }

        this.isTraining = false
        const currentRoundData = this.roundInfo.get('roundData') || { currentRound: this.trainingRound, initiator: this.ydoc.clientID }

        if (currentRoundData.currentRound === this.trainingRound && currentRoundData.status !== 'completed') {

            if (currentRoundData.initiator === this.ydoc.clientID || !currentRoundData.initiator || this.autoTrainingEnabled) {
                this.roundInfo.set('roundData', {
                    ...currentRoundData,
                    status: 'completed',
                    endTime: Date.now(),
                    participantCount: numContributingPeers,
                    aggregatedParameters: aggregatedParams
                })
                
                this.log(`Round ${this.trainingRound} status set to 'completed'.`)

            }
        }

        const localAwarenessState = this.awareness.getLocalState() || {}
        localAwarenessState.round = this.trainingRound
        localAwarenessState.roundStatus = 'completed'
        localAwarenessState.training = false
        this.awareness.setLocalState(localAwarenessState)

        this._dispatchEvent('roundFinalized', {
            round: this.trainingRound,
            participants: numContributingPeers,
            parameters: aggregatedParams
        })

        return true
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
    on(event, callback, options) {
        this.events.addEventListener(event, callback, options)
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
            console.log(`[Decentifai]`, ...args)
        }
    }

    /**
     * Log a warning message if debug is enabled
     * @private
     * @param {...any} args - Arguments to log
     */
    warn(...args) {
        if (this.options.debug) {
            console.warn(`[Decentifai]`, ...args)
        }
    }

    /**
     * Disconnect from the federated learning network and clean up resources.
     */
    disconnect() {
        this.log('Disconnecting from federated learning network...')
        this.autoTrainingEnabled = false
        this.isTraining = false

        if (this.awareness) {
            this.awareness.destroy()
            this.log('Awareness destroyed.')
        }

        if (this.provider) {
            this.provider.disconnect()
            this.provider.destroy()
            this.log('WebRTC provider disconnected and destroyed.')
        }

        this.peers = {}
        this.trainingRound = 0
        this.convergenceHistory = []
        this.convergenceMetrics = { parameterDistance: [], modelLoss: [], trainingAccuracy: [] }
        this.converged = false

        this._dispatchEvent('disconnected', {})
        this.log('Decentifai instance disconnected and cleaned up.')
    }

    /**
     *  Clear all federation history locally (embeddings and parameter updates)
     */
    destroy() {
         if (this.ydoc) {
            this.ydoc.destroy()
            this.log('Y.js document destroyed.')
        }
    }
}