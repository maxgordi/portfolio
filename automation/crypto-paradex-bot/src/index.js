import axios from 'axios';
import { ParadexAuth } from './auth.js';
import { ParadexUtils } from './utils.js';
import dotenv from 'dotenv';

dotenv.config();

// Credentials are read from the environment (.env) only; there are no defaults.
const { ETH_ADDRESS, PARADEX_PRIVATE_KEY, PARADEX_ADDRESS } = process.env;

for (const [name, value] of Object.entries({ ETH_ADDRESS, PARADEX_PRIVATE_KEY, PARADEX_ADDRESS })) {
    if (!value) {
        console.error(`Missing required environment variable: ${name} (see .env.example)`);
        process.exit(1);
    }
}

class ParadexBot {
    constructor() {
        this.auth = new ParadexAuth(ETH_ADDRESS, PARADEX_PRIVATE_KEY, PARADEX_ADDRESS);
        this.initialize();
    }

    async initialize() {
        console.log("🔧 Initializing Paradex Authentication Client");
        console.log("📋 Configuration with Paradex testnet data:");
        console.log("   ETH Address:", this.auth.ethAddress);
        console.log("   Paradex Address:", this.auth.paradexAddress);
        console.log("   Public Key:", this.auth.publicKey);
        console.log("   Private Key:", this.auth.paradexPrivateKey.substring(0, 10) + "...");
    }

    async onboardAccount() {
        try {
            console.log("\n🚀 Step 1: Account Onboarding");
            
            const { chainId } = await ParadexUtils.getSystemConfig();
            console.log("🔗 Chain ID:", chainId);

            const timestamp = Date.now();
            console.log("⏰ Timestamp:", timestamp);

            const signature = this.auth.signOnboardingRequest(chainId);
            const headers = ParadexUtils.createOnboardingHeaders(
                this.auth.ethAddress,
                this.auth.paradexAddress,
                signature,
                timestamp
            );

            const onboardingData = {
                public_key: this.auth.publicKey
            };

            console.log("📤 Sending onboarding request...");
            console.log("📋 Headers:", Object.keys(headers));

            const response = await axios.post(
                `${ParadexUtils.API_BASE_URL}/onboarding`,
                JSON.stringify(onboardingData),
                { headers }
            );

            console.log("✅ Onboarding completed successfully!");
            console.log("📨 Status:", response.status);
            
            if (response.data) {
                console.log("📄 Response data:", response.data);
            }

            return true;

        } catch (error) {
            return ParadexUtils.handleError(error, "onboarding").success;
        }
    }

    async getJWTToken() {
        try {
            console.log("\n🚀 Step 2: Getting JWT token");
            
            const { chainId } = await ParadexUtils.getSystemConfig();
            console.log("🔗 Chain ID:", chainId);

            const timestamp = Math.floor(Date.now() / 1000);
            const expiration = timestamp + 3600;
            
            console.log("⏰ Timestamp:", timestamp);
            console.log("⏰ Expiration:", expiration);

            const signature = this.auth.signAuthRequest(chainId, timestamp, expiration);
            const headers = ParadexUtils.createAuthHeaders(
                this.auth.paradexAddress,
                signature,
                timestamp,
                expiration
            );

            console.log("📤 Sending JWT request...");
            console.log("📋 Headers:", Object.keys(headers));

            const response = await axios.post(
                `${ParadexUtils.API_BASE_URL}/auth`,
                {},
                { headers }
            );

            console.log("✅ JWT token received successfully!");
            console.log("📨 Status:", response.status);
            
            if (response.data?.jwt_token) {
                console.log("🔑 JWT Token:", response.data.jwt_token.substring(0, 50) + "...");
                return response.data.jwt_token;
            }

            console.error("❌ JWT token not found in response");
            console.log("📄 Full response:", response.data);
            return null;

        } catch (error) {
            const result = ParadexUtils.handleError(error, "getting JWT");
            return result.success ? null : "NEEDS_ONBOARDING";
        }
    }

    async authenticate() {
        console.log("🎯 Starting Paradex authentication process");
        console.log("📅 Start time:", new Date().toLocaleString());
        console.log("=".repeat(60));

        try {
            console.log("\n🔄 Attempting onboarding...");
            const onboardingSuccess = await this.onboardAccount();
            
            if (onboardingSuccess) {
                console.log("✅ Onboarding successful!");
            } else {
                console.log("⚠️ Onboarding failed, but continuing with authentication...");
            }
            
            console.log("\n🔄 Attempting to get JWT token...");
            const jwtToken = await this.getJWTToken();
            
            if (jwtToken && jwtToken !== "NEEDS_ONBOARDING") {
                console.log("\n" + "=".repeat(60));
                console.log("🎉 Authentication process completed successfully!");
                console.log("🔐 JWT token received and ready to use");
                console.log("⏰ Token valid for 5 minutes");
                console.log("📅 Completion time:", new Date().toLocaleString());
                console.log("=".repeat(60));
                
                return jwtToken;
            }
            
            throw new Error("Failed to get JWT token after onboarding");
            
        } catch (error) {
            console.log("\n" + "=".repeat(60));
            console.error("💀 Critical error in authentication process:");
            console.error("📄 Message:", error.message);
            console.log("📅 Error time:", new Date().toLocaleString());
            console.log("=".repeat(60));
            
            return null;
        }
    }
}

// Start the bot
const bot = new ParadexBot();
bot.authenticate().then(token => {
    if (token) {
        console.log("\n🔐 Token ready for API calls!");
        console.log("💡 Usage example:");
        console.log("   headers: { 'Authorization': `Bearer ${token}` }");
    } else {
        console.log("\n😞 Authentication failed");
        process.exit(1);
    }
});
