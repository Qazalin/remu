
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4n159>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001718: 86039F0F
	v_mov_b32_e32 v2, 0                                        // 00000000171C: 7E040280
	s_lshl_b64 s[8:9], s[2:3], 2                               // 000000001720: 84888202
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001728: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000172C: 82070907
	s_lshl_b64 s[2:3], s[2:3], 3                               // 000000001730: 84828302
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001734: BF870009
	s_add_u32 s0, s0, s2                                       // 000000001738: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000173C: 82010301
	s_load_b32 s6, s[6:7], null                                // 000000001740: F4000183 F8000000
	s_load_b64 s[0:1], s[0:1], null                            // 000000001748: F4040000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	s_mul_i32 s1, s1, s6                                       // 000000001754: 96010601
	s_mul_hi_u32 s7, s0, s6                                    // 000000001758: 96870600
	s_mul_i32 s0, s0, s6                                       // 00000000175C: 96000600
	s_add_i32 s7, s7, s1                                       // 000000001760: 81070107
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001764: BF870009
	v_dual_mov_b32 v0, s0 :: v_dual_mov_b32 v1, s7             // 000000001768: CA100000 00000007
	s_add_u32 s0, s4, s2                                       // 000000001770: 80000204
	s_addc_u32 s1, s5, s3                                      // 000000001774: 82010305
	global_store_b64 v2, v[0:1], s[0:1]                        // 000000001778: DC6E0000 00000002
	s_nop 0                                                    // 000000001780: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001784: BFB60003
	s_endpgm                                                   // 000000001788: BFB00000
