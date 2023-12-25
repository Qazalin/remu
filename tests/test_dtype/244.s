
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4n106>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001708: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 00000000170C: 86039F0F
	v_mov_b32_e32 v2, 0                                        // 000000001710: 7E040280
	s_lshl_b64 s[8:9], s[2:3], 1                               // 000000001714: 84888102
	s_load_b64 s[0:1], s[0:1], 0x10                            // 000000001718: F4040000 F8000010
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001724: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001728: 82070907
	global_load_u16 v0, v2, s[6:7]                             // 00000000172C: DC4A0000 00060002
	s_lshl_b64 s[6:7], s[2:3], 2                               // 000000001734: 84868202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000001738: BF8700D9
	s_add_u32 s0, s0, s6                                       // 00000000173C: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001740: 82010701
	s_load_b32 s0, s[0:1], null                                // 000000001744: F4000000 F8000000
	s_waitcnt vmcnt(0)                                         // 00000000174C: BF8903F7
	v_bfe_i32 v0, v0, 0, 16                                    // 000000001750: D6110000 02410100
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000001758: 3402009F
	s_waitcnt lgkmcnt(0)                                       // 00000000175C: BF89FC07
	v_add_co_u32 v0, vcc_lo, s0, v0                            // 000000001760: D7006A00 00020000
	s_lshl_b64 s[0:1], s[2:3], 3                               // 000000001768: 84808302
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000176C: BF870009
	s_add_u32 s0, s4, s0                                       // 000000001770: 80000004
	v_add_co_ci_u32_e32 v1, vcc_lo, 0, v1, vcc_lo              // 000000001774: 40020280
	s_addc_u32 s1, s5, s1                                      // 000000001778: 82010105
	global_store_b64 v2, v[0:1], s[0:1]                        // 00000000177C: DC6E0000 00000002
	s_nop 0                                                    // 000000001784: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001788: BFB60003
	s_endpgm                                                   // 00000000178C: BFB00000
