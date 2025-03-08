# TREX-Core Comprehensive Dependency Graph

This document visualizes the complete architecture, dependencies, and interactions within the TREX-Core Transactive Energy simulation framework.

## Core Components and Dependencies

```mermaid
flowchart TB
    %% Main Components
    Runner["Runner\nOrchestrator"] --> SimController["Simulation Controller"]
    Runner --> Market["Market Client"]
    Runner --> Participants["Participant Clients"]
    
    %% Config Dependencies
    Config["JSON Configs"] --> Runner
    Config --> SimController
    Config --> Market
    Config --> Participants
    
    %% MQTT Communication Hub
    MQTT["MQTT Broker"] <--> SimController
    MQTT <--> Market
    MQTT <--> Participants
    
    %% Database Connections
    DB[(Database)] --> Runner
    DB <--> SimController
    DB <--> Market
    DB <--> Participants
    DBUtils["DB Utilities"] --> DB
    
    %% Market Components
    Market --> DoubleAuction["Double Auction\nMarket Mechanism"]
    DoubleAuction --> BidsAsks["Bids and Asks"]
    Market --> Grid["Grid Market"]
    
    %% Participant Components
    Participants --> TraderAgents["Trader Agents"]
    Participants --> DeviceModels["Physical Devices"]
    Participants --> Ledger["Participant Ledger"]
    DeviceModels --> BESS["Battery Storage"]
    
    %% Trader Agent Types
    TraderAgents --> BasicTrader["Basic Trader"]
    TraderAgents --> BaselineAgent["Baseline Agent"]
    TraderAgents --> BatteryScheduleAgent["Battery Schedule Agent"]
    
    %% Trading Logic Flow
    TraderAgents --> BidsAsks
    
    %% Style Settings
    classDef core fill:#f9f,stroke:#333,stroke-width:2px
    classDef component fill:#bbf,stroke:#33f,stroke-width:1px
    classDef data fill:#ffa,stroke:#333,stroke-width:1px
    classDef comms fill:#beb,stroke:#383,stroke-width:1px
    
    class Runner,SimController core
    class Market,Participants,DoubleAuction,TraderAgents,DeviceModels component
    class Config,DB,BidsAsks data
    class MQTT comms
```

## MQTT Communication Pathways

```mermaid
flowchart LR
    %% MQTT Topic Structure
    MQTT((MQTT Broker))
    
    %% Market Topics
    MQTT --> MT1["/market_id"]
    MQTT --> MT2["/market_id/join_market"]
    MQTT --> MT3["/market_id/bid"]
    MQTT --> MT4["/market_id/ask"]
    MQTT --> MT5["/market_id/settlement_delivered"]
    MQTT --> MT6["/market_id/meter"]
    MQTT --> MT7["/market_id/{participant_id}"]
    
    %% Simulation Controller Topics
    MQTT --> ST1["/market_id/simulation/participant_joined"]
    MQTT --> ST2["/market_id/simulation/end_turn"]
    MQTT --> ST3["/market_id/simulation/end_round"]
    MQTT --> ST4["/market_id/simulation/participant_ready"]
    MQTT --> ST5["/market_id/simulation/market_ready"]
    MQTT --> ST6["/market_id/algorithm/policy_sever_ready"]
    MQTT --> ST7["/market_id/simulation/participant_disconnected"]
    MQTT --> ST8["/market_id/start_round"]
    
    %% Components
    Market["Market Client"]
    SimController["Simulation Controller"]
    Participants["Participant Clients"]
    
    %% Market Subscriptions
    MT1 --> Market
    MT2 --> Market
    MT3 --> Market
    MT4 --> Market
    MT5 --> Market
    MT6 --> Market
    ST1 --> Market
    
    %% Simulation Controller Subscriptions
    ST2 --> SimController
    ST3 --> SimController
    ST4 --> SimController
    ST5 --> SimController
    ST6 --> SimController
    ST7 --> SimController
    
    %% Participant Subscriptions
    MT1 --> Participants
    MT7 --> Participants
    ST8 --> Participants
    
    %% Style
    classDef topic fill:#beb,stroke:#383,stroke-width:1px
    classDef component fill:#bbf,stroke:#33f,stroke-width:1px
    
    class MT1,MT2,MT3,MT4,MT5,MT6,MT7,ST1,ST2,ST3,ST4,ST5,ST6,ST7,ST8 topic
    class Market,SimController,Participants component
```

## Data Flow Pathways

```mermaid
flowchart TB
    %% Configuration Flow
    Config["Config Files"] --> Runner
    Runner --> MarketConfig["Market Configuration"]
    Runner --> ParticipantConfig["Participant Configuration"]
    
    %% Market Data Flow
    MarketConfig --> Market["Market Client"]
    Market --> BidAskProcessing["Bid/Ask Processing"]
    BidAskProcessing --> MarketClearing["Market Clearing"]
    MarketClearing --> SettlementProcessing["Settlement Processing"]
    
    %% Participant Data Flow
    ParticipantConfig --> Participant["Participant"]
    ProfileDB[(Profile Database)] --> Participant
    Participant --> LoadGenData["Load/Generation Data"]
    LoadGenData --> TraderAgent["Trader Agent"]
    TraderAgent --> BidAskProcessing
    
    %% Device Control Flow
    TraderAgent --> DeviceControl["Device Control"]
    DeviceControl --> BESS["Battery Storage"]
    BESS --> SettlementProcessing
    
    %% Results Flow
    SettlementProcessing --> OutputDB[(Output Database)]
    
    %% Style Settings
    classDef config fill:#ffa,stroke:#333,stroke-width:1px
    classDef process fill:#bbf,stroke:#33f,stroke-width:1px
    classDef data fill:#fda,stroke:#d50,stroke-width:1px
    classDef db fill:#adf,stroke:#05a,stroke-width:2px
    
    class Config,MarketConfig,ParticipantConfig config
    class Market,BidAskProcessing,MarketClearing,SettlementProcessing,Participant,TraderAgent,DeviceControl,BESS process
    class LoadGenData data
    class ProfileDB,OutputDB db
```

## Execution Flow and Concurrency Model

```mermaid
sequenceDiagram
    participant Runner
    participant SimController as Simulation Controller
    participant Market as Market Client
    participant Participant1 as Participant 1
    participant ParticipantN as Participant N
    participant MQTT as MQTT Broker
    participant DB as Database
    
    Runner->>SimController: Initialize
    Runner->>Market: Initialize
    Runner->>Participant1: Initialize
    Runner->>ParticipantN: Initialize
    
    activate SimController
    activate Market
    activate Participant1
    activate ParticipantN
    
    Note over SimController,ParticipantN: All components run as parallel async tasks
    
    SimController->>MQTT: Subscribe to topics
    Market->>MQTT: Subscribe to topics
    Participant1->>MQTT: Subscribe to topics
    ParticipantN->>MQTT: Subscribe to topics
    
    Note over SimController,ParticipantN: Simulation Start
    
    loop For each time step
        SimController->>MQTT: Publish start_round
        MQTT->>Market: Notify start_round
        MQTT->>Participant1: Notify start_round
        MQTT->>ParticipantN: Notify start_round
        
        Participant1->>Participant1: Process profiles & strategy
        ParticipantN->>ParticipantN: Process profiles & strategy
        
        Participant1->>MQTT: Submit bids/asks
        ParticipantN->>MQTT: Submit bids/asks
        MQTT->>Market: Forward bids/asks
        
        Market->>Market: Process market clearing
        
        Market->>MQTT: Publish settlements
        MQTT->>Participant1: Notify settlements
        MQTT->>ParticipantN: Notify settlements
        
        Participant1->>Participant1: Update devices
        ParticipantN->>ParticipantN: Update devices
        
        Participant1->>MQTT: Confirm settlement delivery
        ParticipantN->>MQTT: Confirm settlement delivery
        MQTT->>Market: Forward confirmations
        
        Participant1->>MQTT: Submit meter data
        ParticipantN->>MQTT: Submit meter data
        MQTT->>Market: Forward meter data
        
        Participant1->>MQTT: Signal participant_ready
        ParticipantN->>MQTT: Signal participant_ready
        MQTT->>SimController: Forward ready signals
        
        Market->>MQTT: Signal market_ready
        MQTT->>SimController: Forward market_ready
        
        SimController->>MQTT: Publish end_round
        MQTT->>Market: Notify end_round
        MQTT->>Participant1: Notify end_round
        MQTT->>ParticipantN: Notify end_round
        
        Market->>DB: Record market data
        Participant1->>DB: Record participant data
        ParticipantN->>DB: Record participant data
    end
    
    deactivate SimController
    deactivate Market
    deactivate Participant1
    deactivate ParticipantN
    
    Runner->>DB: Finalize simulation results
```

## Database Schema and Relationships

```mermaid
classDiagram
    class records {
        +time PK
        +participant_id PK
        +meter JSON
        +remaining_energy Integer
        +state_of_charge Float
    }
    
    class settlements {
        +time Integer
        +bid_id String
        +ask_id String
        +quantity Float
        +price Float
    }
    
    class bids {
        +time Integer
        +bid_id String
        +participant_id String
        +quantity Float
        +price Float
        +delivery_time Tuple
    }
    
    class asks {
        +time Integer
        +ask_id String
        +participant_id String
        +quantity Float
        +price Float
        +delivery_time Tuple
    }
    
    class profiles {
        +time Integer
        +participant_id String
        +load Float
        +generation Float
    }
    
    class configs {
        +config_name String
        +config_data JSON
    }
    
    records --> profiles: references
    settlements --> bids: references
    settlements --> asks: references
    bids --> profiles: relies on
    asks --> profiles: relies on
```

## Double Auction Market Mechanism

```mermaid
flowchart TB
    %% Market Process
    MarketOpen["Market Open"] --> BidsAsks["Collect Bids & Asks"]
    BidsAsks --> MarketClose["Market Close"]
    MarketClose --> SourceClassify["Classify Energy Sources"]
    SourceClassify --> SortBids["Sort Bids by Price"]
    SortBids --> SortAsks["Sort Asks by Price"]
    SortAsks --> MatchOrders["Match Orders"]
    MatchOrders --> DeterminePrice["Determine Settlement Price"]
    DeterminePrice --> NotifyParticipants["Notify Participants"]
    NotifyParticipants --> RecordSettlements["Record Settlements"]
    RecordSettlements --> NextRound["Prepare for Next Round"]
    
    %% Subprocesses
    MatchOrders --> BidExceedsAsk{"Bid Price ≥ Ask Price?"}
    BidExceedsAsk -->|Yes| QuantityMatch["Match Quantities"]
    BidExceedsAsk -->|No| NoMatch["No Match Possible"]
    
    DeterminePrice --> PriceRule{"Price Rule"}
    PriceRule -->|Midpoint| MidpointPrice["Average of Bid & Ask"]
    PriceRule -->|Discriminatory| DiscriminatoryPrice["Bid & Ask Original Prices"]
    PriceRule -->|Uniform| UniformPrice["Market Clearing Price"]
    
    %% Style
    classDef process fill:#bbf,stroke:#33f,stroke-width:1px
    classDef decision fill:#fda,stroke:#d50,stroke-width:1px
    classDef endpoint fill:#beb,stroke:#383,stroke-width:1px
    
    class MarketOpen,BidsAsks,MarketClose,SourceClassify,SortBids,SortAsks,MatchOrders,DeterminePrice,NotifyParticipants,RecordSettlements process
    class BidExceedsAsk,PriceRule decision
    class NextRound,NoMatch,QuantityMatch,MidpointPrice,DiscriminatoryPrice,UniformPrice endpoint
```

## MQTT Message Sequence (Time-Based)

```mermaid
sequenceDiagram
    participant Runner
    participant SimController as Simulation Controller
    participant Market as Market Client
    participant Participant as Participants
    participant MQTT as MQTT Broker
    
    Note over Runner,MQTT: 1. Simulation Initialization
    Runner->>SimController: Start simulation
    SimController->>MQTT: Connect
    Market->>MQTT: Connect
    Participant->>MQTT: Connect
    
    Note over Runner,MQTT: 2. Market Round Start
    SimController->>MQTT: Publish /market_id/start_round
    MQTT->>Market: Forward start_round
    MQTT->>Participant: Forward start_round
    
    Note over Runner,MQTT: 3. Participant Actions
    Participant->>MQTT: Publish /market_id/bid
    Participant->>MQTT: Publish /market_id/ask
    MQTT->>Market: Forward bids
    MQTT->>Market: Forward asks
    
    Note over Runner,MQTT: 4. Market Clearing (Internal)
    Market->>Market: Process bids/asks
    Market->>Market: Determine matches
    
    Note over Runner,MQTT: 5. Settlement Notification
    Market->>MQTT: Publish /market_id/{participant_id}
    MQTT->>Participant: Forward settlements
    
    Note over Runner,MQTT: 6. Settlement Delivery Confirmation
    Participant->>MQTT: Publish /market_id/settlement_delivered
    MQTT->>Market: Forward confirmations
    
    Note over Runner,MQTT: 7. Meter Data Submission
    Participant->>MQTT: Publish /market_id/meter
    MQTT->>Market: Forward meter data
    
    Note over Runner,MQTT: 8. Round End Signal
    Market->>MQTT: Publish /market_id/simulation/market_ready
    MQTT->>SimController: Forward market_ready
    
    Note over Runner,MQTT: 9. Participant Status Updates
    Participant->>MQTT: Publish /market_id/simulation/participant_ready
    MQTT->>SimController: Forward participant_ready
    
    Note over Runner,MQTT: 10. Advance to Next Round
    SimController->>MQTT: Publish /market_id/simulation/end_round
    MQTT->>Market: Forward end_round
    MQTT->>Participant: Forward end_round
```

## Integration with External Systems

```mermaid
flowchart LR
    TREX["TREX-Core System"] <--> CityLearn["CityLearn\nvia configs"]
    TREX <--> PolicyServers["Policy Servers\nvia MQTT"]
    TREX <--> ExternalDB[(External Databases\nvia db_utils)]
    
    %% Style
    classDef core fill:#f9f,stroke:#333,stroke-width:2px
    classDef external fill:#fda,stroke:#d50,stroke-width:1px
    
    class TREX core
    class CityLearn,PolicyServers,ExternalDB external
``` 