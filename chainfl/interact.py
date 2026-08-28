'''
include the upload and the download method for client to interact with the blockchain.
'''
from util import jsonFormat
from collections import defaultdict
# defer brownie imports to runtime (see ensure_connected_and_deployed)
# from brownie import *  # removed for compatibility with environments without brownie installed
# Module-level lazy bindings for Brownie objects, assigned at runtime.
accounts = None
network = None
project = None
Contract = None
import logging

try:
    import torch  # optional, required only for watermark features
except Exception as _e:
    torch = None

logger = logging.getLogger(__name__)

# Lazy blockchain initialization (no import-time side effects)
import os
_project_loaded = False
_connected = False
_deployed = False
_server_accounts = None


def ensure_connected_and_deployed():
    """Idempotently load Brownie project, connect network, and deploy required contracts.
    Avoids import-time side effects and allows offline/testing contexts to import safely.
    """
    global _project_loaded, _connected, _deployed, _server_accounts
    if _deployed:
        return
    try:
        from brownie import project as _project, network as _network, accounts as _accounts, Contract as _Contract  # imported lazily
        # Expose brownie objects at module scope for other functions
        globals()['project'] = _project
        globals()['network'] = _network
        globals()['accounts'] = _accounts
        globals()['Contract'] = _Contract
        # Load project (idempotent)
        try:
            # Find project root directory (where chainEnv is located)
            import pathlib
            current_file = pathlib.Path(__file__).resolve()
            project_root = current_file.parent.parent  # chainfl/interact.py -> PoL-BFL
            chain_env_path = project_root / "chainEnv"

            p = project.load(project_path=str(chain_env_path), name="chainServerZKP")
            p.load_config()
            _project_loaded = True
            logger.debug(f"Brownie project loaded from {chain_env_path}")
        except Exception as e:
            logger.debug(f"project.load skipped/failed: {e}")
        # Import contract wrappers after project load
        from brownie.project.chainServerZKP import watermarkNegotiation, clientManager, PoLContract, AnchorRegistry  # type: ignore
        # Connect to network if not already
        try:
            if not network.is_connected():
                network.connect(os.environ.get('BROWNIE_NETWORK', 'development'))
            _connected = True
        except Exception as e:
            logger.debug(f"network.connect skipped/failed: {e}")
        # Accounts and deployments
        _server_accounts = accounts[0]
        try:
            if len(watermarkNegotiation) == 0:
                watermarkNegotiation.deploy({'from': _server_accounts})
        except Exception as e:
            logger.debug(f"watermarkNegotiation deploy skipped: {e}")
        try:
            if len(clientManager) == 0:
                clientManager.deploy({'from': _server_accounts})
        except Exception as e:
            logger.debug(f"clientManager deploy skipped: {e}")
        try:
            if len(PoLContract) == 0:
                PoLContract.deploy({'from': _server_accounts})
                logger.info("PoLContract deployed successfully")
        except Exception as e:
            logger.debug(f"PoLContract deploy skipped: {e}")
        try:
            if len(AnchorRegistry) == 0:
                AnchorRegistry.deploy({'from': _server_accounts})
                logger.info("AnchorRegistry deployed successfully")
        except Exception as e:
            logger.debug(f"AnchorRegistry deploy skipped: {e}")
        _deployed = True
    except Exception as e:
        logger.warning(f"ensure_connected_and_deployed failed (blockchain offline?): {e}")
        _deployed = False


def upload():
    raise NotImplementedError


#utils for blockchain
#The client communicate with the blockchain through chainProxy.

class chainProxy():
    def __init__(self):
        # Ensure network/project/contracts are ready (no-ops in offline/tests)
        ensure_connected_and_deployed()
        # Import here to avoid import-time hard deps
        try:
            from brownie import accounts
            from brownie.project.chainServerZKP import watermarkNegotiation, PoLContract, AnchorRegistry  # type: ignore
        except Exception as e:
            logger.debug(f"brownie imports unavailable: {e}")
        self.upload_params = None
        try:
            self.account_num = len(accounts) - 1  # accounts used for client
            self.watermark_proxy = watermarkNegotiation[0]
            self.server_accounts = accounts[0]
        except Exception:
            # In offline mode, leave these as None/0
            self.account_num = 0
            self.watermark_proxy = None
            self.server_accounts = None
        self.client_num = 0
        self._issued_challenges = []
        self._resolved_challenges = []

        # PoL contract proxy
        try:
            # Use the most recently deployed PoLContract instance
            try:
                if len(PoLContract) > 0:
                    self.pol_contract = PoLContract[len(PoLContract)-1]
                else:
                    self.pol_contract = None
            except Exception:
                self.pol_contract = None
            # Rebind with latest ABI to ensure new methods available (if available)
            try:
                from brownie import Contract
                from brownie.project import chainServerZKP as _proj_mod
                if self.pol_contract is not None:
                    self.pol_contract = Contract.from_abi("PoLContract", self.pol_contract.address, _proj_mod.PoLContract._build['abi'])
            except Exception as e:
                logger.debug(f"PoL ABI rebind skipped: {e}")
            logger.info("PoL contract proxy initialized")
        except Exception as e:
            self.pol_contract = None
            logger.warning(f"PoL contract not available: {e}")

        # Optional: initialize on-chain Groth16 Verifier (if compiled in project)
        self.zkp_verifier_contract = None
        try:
            # Contract class name exported by snarkjs is typically `Verifier`
            VerifierClass = None
            try:
                VerifierClass = Verifier  # type: ignore[name-defined]
            except NameError:
                try:
                    VerifierClass = Groth16Verifier  # type: ignore[name-defined]
                except NameError:
                    VerifierClass = None
            if VerifierClass is not None:
                # Always deploy a fresh verifier on current dev network to avoid stale-address issues
                try:
                    self.zkp_verifier_contract = VerifierClass.deploy({'from': self.server_accounts})
                    logger.info("Groth16 Verifier deployed successfully (fresh)")
                except Exception as e:
                    logger.debug(f"Groth16 Verifier deploy skipped: {e}")
                    self.zkp_verifier_contract = None
            else:
                logger.debug("Groth16 Verifier class not found in project; skip on-chain ZKP verify init")
        except Exception as e:
            logger.debug(f"Groth16 Verifier init skipped: {e}")

        # Bind verifier to PoL contract (disabled by default)
        try:
            if self.pol_contract is not None and self.zkp_verifier_contract is not None:
                setv = getattr(self.pol_contract, 'setVerifier', None) or self.pol_contract.get_method('setVerifier')
                setv(self.zkp_verifier_contract.address, False, {'from': self.server_accounts})
        except Exception as e:
            logger.debug(f"Set verifier on PoLContract skipped: {e}")

        # AnchorRegistry contract proxy
        self.anchor_registry = None
        try:
            if len(AnchorRegistry) > 0:
                self.anchor_registry = AnchorRegistry[len(AnchorRegistry)-1]
                logger.info("AnchorRegistry contract proxy initialized")
        except Exception as e:
            logger.warning(f"AnchorRegistry contract not available: {e}")

        '''
        Here Brownie store all address in a vector.
        We just delegate the index of the vector to the client as the ClientID (str)
        The interaction with the blockchain is mainly through the ethereum.
        '''
        self.client_list = defaultdict(type(accounts[0].address))
        # blockchain_init
    def get_account_num(self):
        return self.account_num
    def get_client_num(self):
        return self.client_num

    def get_client_list(self):
        return self.client_list


    def add_account(self)->str:
        account = accounts.add()
        self.account_num += 1
        return account.address

    #construct the projection between account and client
    def client_regist(self)->str:
        self.client_num += 1
        if(self.account_num<self.client_num):
            self.add_account()
        self.client_list[str(self.client_num)] = accounts[self.client_num]
        return str(self.client_num)

    def watermark_negotitaion(self,client_id:str,watermark_length=64):
        client_id_int = self._extract_client_id_int(client_id)
        self.watermark_proxy.generate_watermark({'from':accounts[client_id_int]})

    def upload_model(self,upload_params:dict):
        '''
        This function recieve a dict and the value in this dict must be the type which json can serilized
        And there must have a key named state_dict and the value type is OrderedDict in pytorch model.state_dict()
        This function will turn state_dict into list, so the user dont need to turn into list at first.
        '''
        model_state_dict = upload_params['state_dict']
        upload_params['state_dict'] = jsonFormat.model2json(model_state_dict)
        #Upload
        self.upload_params = upload_params
        return

    def download_model(self, params = None):
        '''
        从区块链上接受json格式的字符串为全局模型并下载。
        但会返回一个orderdict作为全局模型的state_dict
        '''
        download_params = self.upload_params
        download_params['state_dict']  = jsonFormat.json2model(download_params['state_dict'])
        return download_params

    def construct_sign(self, args: dict = {}):
        sign_config = args.get('sign_config')
        model_name  = args.get('model')
        bit_length  = args.get('bit_length')

        if model_name != "SignAlexNet":
            logger.error("Watermark Not Support for this network")
            raise Exception("Watermark Not Support for this network")

        watermark_args = dict()
        alexnet_channels = {
        '4': (384, 3456),
        '5': (256, 2304),
        '6': (256, 2304)
        }

        for layer_key in sign_config:
            flag = sign_config[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            watermark_args[layer_key] = {
                'flag': flag
            }

            if b is not None:
                if layer_key == "4":
                    output_channels = int (bit_length * 384 / 896)
                if layer_key == "5":
                    output_channels = int (bit_length * 256/ 896)
                if layer_key == "6":
                    output_channels = int (bit_length * 256/ 896)

                if torch is None:
                    logger.error("Torch is required for watermark sign generation; please install torch or disable watermark features.")
                    raise RuntimeError("torch not available for watermark sign generation")
                b = torch.sign(torch.rand(output_channels) - 0.5)
                M = torch.randn(alexnet_channels[layer_key][0], output_channels)

                watermark_args[layer_key]['b'] = b
                watermark_args[layer_key]['M'] = M

        return watermark_args

    # ========== PoL相关方法 ==========

    def pol_register_client(self, client_id: str) -> bool:
        """
        在PoL合约中注册客户端

        Args:
            client_id: 客户端ID（支持"client_X"或"X"格式）

        Returns:
            success: 是否成功
        """
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return False

        try:
            # 提取数字ID（支持"client_X"或"X"格式）
            client_id_int = self._extract_client_id_int(client_id)
            client_account = accounts[client_id_int]

            # 调用合约的registerClient方法
            tx = self.pol_contract.registerClient({'from': client_account})
            logger.info(f"Client {client_id} registered in PoL contract")
            logger.debug(f"  Transaction: {tx.txid}")
            return True
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            return False

    def submit_pol_proof(self, client_id: str, commitment: str,
                        data_hash: str, num_checkpoints: int,
                        total_steps: int) -> str:
        """
        提交PoL证明到区块链

        Args:
            client_id: 客户端ID
            commitment: Merkle root（十六进制字符串）
            data_hash: 数据哈希（十六进制字符串）
            num_checkpoints: checkpoint数量
            total_steps: 总步数

        Returns:
            tx_hash: 交易哈希
        """
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""

        try:
            client_id_int = self._extract_client_id_int(client_id)
            client_account = accounts[client_id_int]

            # 将十六进制字符串转换为bytes32
            commitment_bytes = bytes.fromhex(commitment.replace('0x', ''))
            data_hash_bytes = bytes.fromhex(data_hash.replace('0x', ''))

            # 调用合约的submitProof方法
            tx = self.pol_contract.submitProof(
                commitment_bytes,
                data_hash_bytes,
                num_checkpoints,
                total_steps,
                {'from': client_account}
            )

            logger.info(f"PoL proof submitted for client {client_id}")
            logger.debug(f"  Commitment: {commitment[:16]}...")
            logger.debug(f"  Transaction: {tx.txid}")

            return tx.txid
        except Exception as e:
            logger.error(f"Failed to submit PoL proof: {e}")
            return ""

    def record_pol_verification(self, client_id: str, is_valid: bool) -> bool:
        """
        记录PoL验证结果（仅服务器可调用）

        Args:
            client_id: 客户端ID
            is_valid: 验证结果

        Returns:
            success: 是否成功
        """
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return False

        try:
            client_id_int = self._extract_client_id_int(client_id)
            client_address = accounts[client_id_int].address

            # 调用合约的recordVerification方法（使用服务器账户）
            tx = self.pol_contract.recordVerification(
                client_address,
                is_valid,
                {'from': self.server_accounts}
            )

            logger.info(f"Verification recorded for client {client_id}: {is_valid}")
            logger.debug(f"  Transaction: {tx.txid}")
            return True
        except Exception as e:
            logger.error(f"Failed to record verification: {e}")
            return False

    def batch_record_pol_verification(self, client_ids: list, results: list) -> bool:
        """
        批量记录PoL验证结果

        Args:
            client_ids: 客户端ID列表
            results: 验证结果列表

        Returns:
            success: 是否成功
        """
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return False

        try:
            # 转换客户端ID为地址（支持"client_X"或"X"格式）
            client_addresses = []
            for cid in client_ids:
                client_id_int = self._extract_client_id_int(cid)
                client_addresses.append(accounts[client_id_int].address)

            # 调用合约的batchRecordVerification方法
            tx = self.pol_contract.batchRecordVerification(
                client_addresses,
                results,
                {'from': self.server_accounts}
            )

            logger.info(f"Batch verification recorded for {len(client_ids)} clients")
            logger.debug(f"  Transaction: {tx.txid}")
            return True
        except Exception as e:
            logger.error(f"Failed to batch record verification: {e}")
            return False

    def get_pol_proof(self, client_id: str) -> dict:
        """
        获取客户端的PoL证明

        Args:
            client_id: 客户端ID

        Returns:
            proof: 证明数据字典
        """
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return {}

        try:
            client_id_int = self._extract_client_id_int(client_id)
            client_address = accounts[client_id_int].address

            # 调用合约的getProof方法
            proof_data = self.pol_contract.getProof(client_address)

            proof = {
                'commitment': proof_data[0].hex(),
                'data_hash': proof_data[1].hex(),
                'num_checkpoints': proof_data[2],
                'total_steps': proof_data[3],
                'timestamp': proof_data[4],
                'verified': proof_data[5],
                'is_valid': proof_data[6]
            }

            return proof
        except Exception as e:
            logger.error(f"Failed to get PoL proof: {e}")
            return {}

    def get_pol_stats(self) -> dict:
        """
        获取PoL合约统计信息

        Returns:
            stats: 统计信息字典
        """
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return {}

        try:
            stats_data = self.pol_contract.getStats()

            stats = {
                'total_proofs': stats_data[0],
                'total_verifications': stats_data[1],
                'total_clients': stats_data[2]
            }

            return stats
        except Exception as e:
            logger.error(f"Failed to get PoL stats: {e}")
            return {}

    # ========== Economic incentive wrappers ==========

    def _extract_client_id_int(self, client_id: str) -> int:
        """
        提取客户端ID的整数部分（支持"client_X"或"X"格式）

        Args:
            client_id: 客户端ID（"client_X"或"X"格式）

        Returns:
            client_id_int: 整数ID
        """
        if isinstance(client_id, int):
            return client_id
        if client_id.startswith('client_'):
            return int(client_id.split('_')[1])
        return int(client_id)

    def _to_address(self, client_id: str):
        client_id_int = self._extract_client_id_int(client_id)
        return accounts[client_id_int].address

    def stake(self, client_id: str, amount_wei: int) -> str:
        """Client stakes ETH (wei) into PoL contract."""
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            client_id_int = self._extract_client_id_int(client_id)
            tx = self.pol_contract.stake({'from': accounts[client_id_int], 'value': int(amount_wei)})
            logger.info(f"Client {client_id} staked {amount_wei} wei")
            return tx.txid
        except Exception as e:
            logger.error(f"Stake failed for {client_id}: {e}")
            return ""

    def unstake(self, client_id: str, amount_wei: int) -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            client_id_int = self._extract_client_id_int(client_id)
            tx = self.pol_contract.unstake(int(amount_wei), {'from': accounts[client_id_int]})
            logger.info(f"Client {client_id} unstaked {amount_wei} wei")
            return tx.txid
        except Exception as e:
            logger.error(f"Unstake failed for {client_id}: {e}")
            return ""

    def lock_stake(self, client_id: str, amount_wei: int) -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            addr = self._to_address(client_id)
            tx = self.pol_contract.lockStake(addr, int(amount_wei), {'from': self.server_accounts})
            logger.info(f"Locked {amount_wei} wei for {client_id}")
            return tx.txid
        except Exception as e:
            logger.error(f"Lock stake failed for {client_id}: {e}")
            return ""

    def unlock_stake(self, client_id: str, amount_wei: int) -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            addr = self._to_address(client_id)
            tx = self.pol_contract.unlockStake(addr, int(amount_wei), {'from': self.server_accounts})
            logger.info(f"Unlocked {amount_wei} wei for {client_id}")
            return tx.txid
        except Exception as e:
            logger.error(f"Unlock stake failed for {client_id}: {e}")
            return ""

    def penalize(self, client_id: str, amount_wei: int, reason: str = "verification_failed") -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            addr = self._to_address(client_id)
            tx = self.pol_contract.penalize(addr, int(amount_wei), reason, {'from': self.server_accounts})
            logger.warning(f"Penalized {client_id} amount={amount_wei} wei reason={reason}")
            return tx.txid
        except Exception as e:
            logger.error(f"Penalize failed for {client_id}: {e}")
            return ""

    def distribute_reward(self, client_id: str, amount_wei: int) -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            addr = self._to_address(client_id)
            tx = self.pol_contract.distributeReward(addr, int(amount_wei), {'from': self.server_accounts})
            logger.info(f"Distributed reward to {client_id}: {amount_wei} wei")
            return tx.txid
        except Exception as e:
            logger.error(f"Distribute reward failed for {client_id}: {e}")
            return ""

    def distribute_rewards(self, client_ids: list, amounts_wei: list) -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            addrs = [self._to_address(cid) for cid in client_ids]
            amts = [int(a) for a in amounts_wei]
            tx = self.pol_contract.batchDistributeRewards(addrs, amts, {'from': self.server_accounts})
            logger.info(f"Batch distributed rewards to {len(addrs)} clients")
            return tx.txid
        except Exception as e:
            logger.error(f"Batch distribute rewards failed: {e}")
            return ""

    def update_reputation(self, client_id: str, score_int: int) -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            addr = self._to_address(client_id)
            tx = self.pol_contract.updateReputation(addr, int(score_int), {'from': self.server_accounts})
            logger.info(f"Updated reputation for {client_id} -> {score_int}")
            return tx.txid
        except Exception as e:
            logger.error(f"Update reputation failed for {client_id}: {e}")
            return ""

    def fund_reward_pool(self, amount_wei: int) -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            tx = self.pol_contract.fundRewardPool({'from': self.server_accounts, 'value': int(amount_wei)})
            logger.info(f"Funded reward pool with {amount_wei} wei")
            return tx.txid
        except Exception as e:
            logger.error(f"Fund reward pool failed: {e}")
            return ""

    def get_stake_info(self, client_id: str) -> dict:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return {}
        try:
            addr = self._to_address(client_id)
            info = self.pol_contract.getStakeInfo(addr)
            return {
                'total': int(info[0]),
                'locked': int(info[1]),
                'available': int(info[2])
            }
        except Exception as e:
            logger.error(f"Get stake info failed for {client_id}: {e}")
            return {}

    def get_reputation(self, client_id: str) -> int:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return 0
        try:
            addr = self._to_address(client_id)
            return int(self.pol_contract.getReputation(addr))
        except Exception as e:
            logger.error(f"Get reputation failed for {client_id}: {e}")
            return 0

    def get_incentive_stats(self) -> dict:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return {}
        try:
            stats = self.pol_contract.getIncentiveStats()
            return {
                'reward_pool': int(stats[0]),
                'penalty_pool': int(stats[1]),
                'total_staked': int(stats[2]),
                'min_stake': int(stats[3])
            }
        except Exception as e:
            logger.error(f"Get incentive stats failed: {e}")
            return {}


    # ========== Challenge/Proof wrappers ==========
    def issue_challenge(self, client_id: str, idx0: int, idx1: int, deadline_ts: int) -> str:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return ""
        try:
            addr = self._to_address(client_id)
            method = getattr(self.pol_contract, 'issueChallenge', None) or self.pol_contract.get_method('issueChallenge')
            tx = method(addr, int(idx0), int(idx1), int(deadline_ts), {'from': self.server_accounts})
            cid = None
            try:
                if 'ChallengeIssued' in tx.events:
                    ev = tx.events['ChallengeIssued']
                    if isinstance(ev, list) and len(ev) > 0:
                        cid = ev[0]['challengeId']
                    else:
                        cid = ev['challengeId']
            except Exception as e:
                logger.debug(f"Event parse skipped: {e}")
            if cid is None:
                cid = getattr(tx, 'return_value', None)
            cid_hex = cid.hex() if hasattr(cid, 'hex') else (cid if isinstance(cid, str) else str(cid))
            if isinstance(cid_hex, str) and not cid_hex.startswith('0x'):
                cid_hex = '0x' + cid_hex
            try:
                self._issued_challenges.append(cid_hex)
            except Exception as e:
                logger.debug(f"Issued challenges tracking skipped: {e}")
            return cid_hex
        except Exception as e:
            logger.error(f"Issue challenge failed: {e}")
            # Optional offline fallback (guarded by env flag)
            import os
            if os.environ.get('POL_OFFLINE_FALLBACK', '0') == '1':
                try:
                    import time
                    cid_hex = f"offline-{int(time.time()*1000)}"
                    try:
                        self._issued_challenges.append(cid_hex)
                    except Exception as e:
                        logger.debug(f"Offline issued challenge tracking skipped: {e}")
                    logger.warning(f"Offline challenge recorded id={cid_hex}")
                    return cid_hex
                except Exception:
                    return ""
            return ""

    def challenge_proof(self, challenge_id_hex: str, public_signals: dict, verified: bool, reason: str = "") -> bool:
        if self.pol_contract is None:
            logger.error("PoL contract not available")
            return False
        try:
            # Optional offline path (guarded by env flag)
            import os
            if not (challenge_id_hex.startswith('0x') and len(challenge_id_hex) > 2):
                if os.environ.get('POL_OFFLINE_FALLBACK', '0') == '1':
                    try:
                        self._resolved_challenges.append((challenge_id_hex, bool(verified)))
                    except Exception as e:
                        logger.debug(f"Offline resolved challenge tracking skipped: {e}")
                    logger.warning(f"Offline challenge resolved id={challenge_id_hex} success={verified}")
                    return True
                return False

            # On-chain path
            if challenge_id_hex.startswith('0x'):
                cid_bytes = bytes.fromhex(challenge_id_hex[2:])
            else:
                cid_bytes = bytes.fromhex(challenge_id_hex)
            W_t_hash = int(public_signals.get('W_t_hash', 0))
            W_t1_hash = int(public_signals.get('W_t1_hash', 0))
            data_hash = int(public_signals.get('data_hash', 0))
            method = getattr(self.pol_contract, 'challengeProof', None) or self.pol_contract.get_method('challengeProof')
            tx = method(cid_bytes, W_t_hash, W_t1_hash, data_hash, bool(verified), reason, {'from': self.server_accounts})
            try:
                self._resolved_challenges.append((challenge_id_hex, bool(verified)))
            except Exception as e:
                logger.debug(f"On-chain resolved challenge tracking skipped: {e}")
            logger.info(f"Challenge resolved id={challenge_id_hex} success={verified}")
            logger.debug(f"  tx: {tx.txid}")
            return True
        except Exception as e:
            logger.error(f"Challenge proof submission failed: {e}")
            return False

    def set_onchain_verifier_enabled(self, use: bool) -> bool:
        if self.pol_contract is None:
            return False
        def _ensure_verifier():
            if getattr(self, 'zkp_verifier_contract', None) is None:
                try:
                    VerifierClass = None
                    try:
                        VerifierClass = Verifier  # type: ignore[name-defined]
                    except NameError:
                        try:
                            VerifierClass = Groth16Verifier  # type: ignore[name-defined]
                        except NameError:
                            VerifierClass = None
                    if VerifierClass is not None:
                        self.zkp_verifier_contract = VerifierClass.deploy({'from': self.server_accounts})
                except Exception as ee:
                    logger.debug(f"Ensure verifier failed: {ee}")
            return getattr(self, 'zkp_verifier_contract', None) is not None
        try:
            if not _ensure_verifier():
                return False
            setv = getattr(self.pol_contract, 'setVerifier', None) or self.pol_contract.get_method('setVerifier')
            setv(self.zkp_verifier_contract.address, bool(use), {'from': self.server_accounts})
            return True
        except Exception as e:
            # try once more with fresh deploy
            self.zkp_verifier_contract = None
            if not _ensure_verifier():
                logger.error(f"Toggle on-chain verifier failed: {e}")
                return False
            try:
                setv = getattr(self.pol_contract, 'setVerifier', None) or self.pol_contract.get_method('setVerifier')
                setv(self.zkp_verifier_contract.address, bool(use), {'from': self.server_accounts})
                return True
            except Exception as e2:
                logger.error(f"Toggle on-chain verifier failed after redeploy: {e2}")
                return False

    def challenge_proof_with_zkp_onchain(self, challenge_id_hex: str, proof: dict, public_signals: dict, reason: str = "") -> bool:
        if self.pol_contract is None or self.zkp_verifier_contract is None:
            logger.error("Contracts not available")
            return False
        try:
            # bytes32 id
            cid_bytes = bytes.fromhex(challenge_id_hex[2:] if challenge_id_hex.startswith('0x') else challenge_id_hex)
            # map proof
            pi_a = proof.get('pi_a') or proof.get('A')
            pi_b = proof.get('pi_b') or proof.get('B')
            pi_c = proof.get('pi_c') or proof.get('C')
            a = [self._intify(pi_a[0]), self._intify(pi_a[1])]
            b = [[self._intify(pi_b[0][1]), self._intify(pi_b[0][0])],
                 [self._intify(pi_b[1][1]), self._intify(pi_b[1][0])]]
            c = [self._intify(pi_c[0]), self._intify(pi_c[1])]
            inputs = [
                self._intify(public_signals.get('W_t_hash', 0)),
                self._intify(public_signals.get('W_t1_hash', 0)),
                self._intify(public_signals.get('data_hash', 0)),
                self._intify(public_signals.get('max_distance', 0)),
            ]
            method = getattr(self.pol_contract, 'challengeProofOnchainVerify', None) or self.pol_contract.get_method('challengeProofOnchainVerify')
            tx = method(cid_bytes, a, b, c, inputs, reason, {'from': self.server_accounts})
            logger.info(f"On-chain integrated ZKP verification tx={tx.txid}")
            return True
        except Exception as e:
            logger.error(f"On-chain integrated ZKP verification failed: {e}")
            return False


    def challenge_proof_with_zkp_onchain_receipt(self, challenge_id_hex: str, proof: dict, public_signals: dict, reason: str = ""):
        """Same as challenge_proof_with_zkp_onchain but returns tx receipt for gas/time measurement.
        Returns tx object on success; None on failure.
        """
        if self.pol_contract is None or getattr(self, 'zkp_verifier_contract', None) is None:
            logger.error("Contracts not available")
            return None
        try:
            cid_bytes = bytes.fromhex(challenge_id_hex[2:] if challenge_id_hex.startswith('0x') else challenge_id_hex)
            pi_a = proof.get('pi_a') or proof.get('A')
            pi_b = proof.get('pi_b') or proof.get('B')
            pi_c = proof.get('pi_c') or proof.get('C')
            a = [self._intify(pi_a[0]), self._intify(pi_a[1])]
            b = [[self._intify(pi_b[0][1]), self._intify(pi_b[0][0])],
                 [self._intify(pi_b[1][1]), self._intify(pi_b[1][0])]]
            c = [self._intify(pi_c[0]), self._intify(pi_c[1])]
            inputs = [
                self._intify(public_signals.get('W_t_hash', 0)),
                self._intify(public_signals.get('W_t1_hash', 0)),
                self._intify(public_signals.get('data_hash', 0)),
                self._intify(public_signals.get('max_distance', 0)),
            ]
            method = getattr(self.pol_contract, 'challengeProofOnchainVerify', None) or self.pol_contract.get_method('challengeProofOnchainVerify')
            tx = method(cid_bytes, a, b, c, inputs, reason, {'from': self.server_accounts})
            logger.info(f"Integrated ZKP tx: {tx.txid} gas_used={getattr(tx, 'gas_used', 'n/a')}")
            return tx
        except Exception as e:
            logger.error(f"On-chain integrated ZKP verification (receipt) failed: {e}")
            return None

    def get_challenge(self, challenge_id_hex: str) -> dict:
        if self.pol_contract is None:
            return {}
        try:
            # Optional offline path (guarded by env flag)
            import os
            if not (challenge_id_hex.startswith('0x') and len(challenge_id_hex) > 2):
                if os.environ.get('POL_OFFLINE_FALLBACK', '0') == '1':
                    resolved = any(cid == challenge_id_hex for cid, _ in getattr(self, '_resolved_challenges', []))
                    return {
                        'client': '0x0000000000000000000000000000000000000000',
                        'idx0': 0,
                        'idx1': 1,
                        'issuedAt': 0,
                        'deadline': 0,
                        'resolved': bool(resolved),
                        'success': True,
                        'reason': 'offline',
                        'W_t_hash': 0,
                        'W_t1_hash': 0,
                        'data_hash': 0,
                    }
                return {}



                return {}
            # On-chain path
            if challenge_id_hex.startswith('0x'):
                cid_bytes = bytes.fromhex(challenge_id_hex[2:])
            else:
                cid_bytes = bytes.fromhex(challenge_id_hex)
            res = self.pol_contract.getChallenge(cid_bytes)
            return {
                'client': res[0],
                'idx0': int(res[1]),
                'idx1': int(res[2]),
                'issuedAt': int(res[3]),
                'deadline': int(res[4]),
                'resolved': bool(res[5]),
                'success': bool(res[6]),
                'reason': str(res[7]),
                'W_t_hash': int(res[8]),
                'W_t1_hash': int(res[9]),
                'data_hash': int(res[10]),
            }
        except Exception as e:
            logger.error(f"Get challenge failed: {e}")
            return {}

    # ========== On-chain Groth16 verification ==========
    def _intify(self, x):
        try:
            if isinstance(x, str) and x.startswith('0x'):
                return int(x, 16)
            return int(x)
        except Exception:
            return int(x)  # let it raise if truly invalid

    def verify_zkp_onchain(self, proof: dict, public_signals: dict) -> bool:
        """Verify Groth16 proof on-chain via Verifier.verifyProof.
        Returns True if verification succeeds on-chain, False otherwise.
        """
        def _ensure_verifier():
            if getattr(self, 'zkp_verifier_contract', None) is None:
                try:
                    VerifierClass = None
                    try:
                        VerifierClass = Verifier  # type: ignore[name-defined]
                    except NameError:
                        try:
                            VerifierClass = Groth16Verifier  # type: ignore[name-defined]
                        except NameError:
                            VerifierClass = None
                    if VerifierClass is not None:
                        try:
                            self.zkp_verifier_contract = VerifierClass.deploy({'from': self.server_accounts})
                            logger.info("Groth16 Verifier deployed successfully (fresh)")
                        except Exception as e:
                            logger.debug(f"Groth16 Verifier deploy skipped: {e}")
                            self.zkp_verifier_contract = None
                except Exception as e:
                    logger.debug(f"Ensure verifier failed: {e}")
            return getattr(self, 'zkp_verifier_contract', None) is not None

        if not _ensure_verifier():
            logger.error("Groth16 Verifier contract not available; ensure contracts are compiled and deployed")
            return False
        try:
            # Expect proof fields compatible with snarkjs output
            pi_a = proof.get('pi_a') or proof.get('A')
            pi_b = proof.get('pi_b') or proof.get('B')
            pi_c = proof.get('pi_c') or proof.get('C')
            assert pi_a is not None and pi_b is not None and pi_c is not None, "Invalid proof fields"

            a = [self._intify(pi_a[0]), self._intify(pi_a[1])]
            # Note: snarkjs outputs pi_b with coordinates that require swapping for Solidity verifier
            b = [
                [self._intify(pi_b[0][1]), self._intify(pi_b[0][0])],
                [self._intify(pi_b[1][1]), self._intify(pi_b[1][0])],
            ]
            c = [self._intify(pi_c[0]), self._intify(pi_c[1])]

            # Order of inputs must match the circuit public signals order
            inputs = [
                self._intify(public_signals.get('W_t_hash', 0)),
                self._intify(public_signals.get('W_t1_hash', 0)),
                self._intify(public_signals.get('data_hash', 0)),
                self._intify(public_signals.get('max_distance', 0)),
            ]

            # Verifier.verifyProof is a view function; brownie will make a call
            ok = bool(self.zkp_verifier_contract.verifyProof(a, b, c, inputs, {'from': self.server_accounts}))
            logger.info(f"On-chain ZKP verification result: {ok}")
            return ok
        except Exception as e:
            # Try once to redeploy fresh and retry (handles network resets)
            logger.debug(f"Verifier call failed, will try redeploy once: {e}")
            self.zkp_verifier_contract = None
            if not _ensure_verifier():
                logger.error(f"On-chain ZKP verification failed: {e}")
                return False
            try:
                pi_a = proof.get('pi_a') or proof.get('A')
                pi_b = proof.get('pi_b') or proof.get('B')
                pi_c = proof.get('pi_c') or proof.get('C')
                a = [self._intify(pi_a[0]), self._intify(pi_a[1])]
                b = [[self._intify(pi_b[0][1]), self._intify(pi_b[0][0])],
                     [self._intify(pi_b[1][1]), self._intify(pi_b[1][0])]]
                c = [self._intify(pi_c[0]), self._intify(pi_c[1])]
                inputs = [self._intify(public_signals.get('W_t_hash', 0)),
                          self._intify(public_signals.get('W_t1_hash', 0)),
                          self._intify(public_signals.get('data_hash', 0)),
                          self._intify(public_signals.get('max_distance', 0))]
                ok = bool(self.zkp_verifier_contract.verifyProof(a, b, c, inputs, {'from': self.server_accounts}))
                logger.info(f"On-chain ZKP verification result (after redeploy): {ok}")
                return ok
            except Exception as e2:
                logger.error(f"On-chain ZKP verification failed after redeploy: {e2}")
                return False

    def anchor_round(self, round_id: str, commit_hash_hex: str, sigset_hash_hex: str) -> dict:
        """
        Anchor a verification round on-chain via AnchorRegistry contract.

        Args:
            round_id: Round identifier (string, will be hashed to bytes32)
            commit_hash_hex: Hex string of commit hash (SHA256 of client commitments)
            sigset_hash_hex: Hex string of sigset hash (SHA256 of verifier addresses)

        Returns:
            dict: {'txid': transaction_hash, 'blockNumber': block_number}

        Raises:
            Exception: If AnchorRegistry not available or transaction fails
        """
        if self.anchor_registry is None:
            raise RuntimeError("AnchorRegistry contract not available; ensure it is deployed")

        try:
            import hashlib
            from brownie import web3

            # Convert round_id to bytes32
            if isinstance(round_id, str):
                round_id_bytes32 = web3.keccak(text=round_id)
            else:
                round_id_bytes32 = round_id

            # Convert hex strings to bytes32
            if commit_hash_hex.startswith('0x'):
                commit_hash_hex = commit_hash_hex[2:]
            if sigset_hash_hex.startswith('0x'):
                sigset_hash_hex = sigset_hash_hex[2:]

            commit_hash_bytes32 = bytes.fromhex(commit_hash_hex)
            sigset_hash_bytes32 = bytes.fromhex(sigset_hash_hex)

            # Call anchorRound on contract
            tx = self.anchor_registry.anchorRound(
                round_id_bytes32,
                commit_hash_bytes32,
                sigset_hash_bytes32,
                {'from': self.server_accounts}
            )

            logger.info(f"Round anchored on-chain: txid={tx.txid}, block={tx.block_number}")

            return {
                'txid': tx.txid,
                'blockNumber': tx.block_number
            }
        except Exception as e:
            logger.error(f"anchor_round failed: {e}")
            raise


class _LazyProxy:
    _inst = None
    def _ensure(self):
        if self._inst is None:
            try:
                self._inst = chainProxy()
            except Exception as e:
                logger.warning(f"chainProxy init failed (lazy): {e}")
                raise
        return self._inst
    def __getattr__(self, name):
        return getattr(self._ensure(), name)

# Export a lazy proxy to avoid import-time side effects
chain_proxy = _LazyProxy()
