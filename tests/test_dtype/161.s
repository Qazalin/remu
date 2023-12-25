
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_2_2n8>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshl_b32 s2, s15, 1                                      // 000000001714: 8402810F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001718: BF870499
	s_ashr_i32 s3, s2, 31                                      // 00000000171C: 86039F02
	s_lshl_b64 s[12:13], s[2:3], 3                             // 000000001720: 848C8302
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s2, s6, s12                                      // 000000001728: 80020C06
	s_addc_u32 s3, s7, s13                                     // 00000000172C: 82030D07
	s_lshl_b32 s6, s14, 1                                      // 000000001730: 8406810E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_ashr_i32 s7, s6, 31                                      // 000000001738: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 3                               // 00000000173C: 84868306
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001740: BF870009
	s_add_u32 s6, s0, s6                                       // 000000001744: 80060600
	s_addc_u32 s7, s1, s7                                      // 000000001748: 82070701
	s_load_b128 s[0:3], s[2:3], null                           // 00000000174C: F4080001 F8000000
	s_load_b128 s[8:11], s[6:7], null                          // 000000001754: F4080203 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000175C: BF89FC07
	s_mul_i32 s1, s8, s1                                       // 000000001760: 96010108
	s_mul_hi_u32 s6, s8, s0                                    // 000000001764: 96860008
	s_mul_i32 s7, s9, s0                                       // 000000001768: 96070009
	s_mul_i32 s0, s8, s0                                       // 00000000176C: 96000008
	s_mul_i32 s3, s10, s3                                      // 000000001770: 9603030A
	s_mul_hi_u32 s8, s10, s2                                   // 000000001774: 9688020A
	s_mul_i32 s9, s11, s2                                      // 000000001778: 9609020B
	s_add_i32 s1, s6, s1                                       // 00000000177C: 81010106
	s_add_i32 s3, s8, s3                                       // 000000001780: 81030308
	s_mul_i32 s2, s10, s2                                      // 000000001784: 9602020A
	s_add_i32 s1, s1, s7                                       // 000000001788: 81010701
	s_add_i32 s3, s3, s9                                       // 00000000178C: 81030903
	s_add_u32 s0, s2, s0                                       // 000000001790: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001794: 82010103
	s_add_u32 s4, s4, s12                                      // 000000001798: 80040C04
	s_addc_u32 s5, s5, s13                                     // 00000000179C: 82050D05
	s_ashr_i32 s15, s14, 31                                    // 0000000017A0: 860F9F0E
	v_mov_b32_e32 v0, s0                                       // 0000000017A4: 7E000200
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, s1              // 0000000017A8: CA100080 02000001
	s_lshl_b64 s[2:3], s[14:15], 3                             // 0000000017B0: 8482830E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B4: BF870009
	s_add_u32 s0, s4, s2                                       // 0000000017B8: 80000204
	s_addc_u32 s1, s5, s3                                      // 0000000017BC: 82010305
	global_store_b64 v2, v[0:1], s[0:1]                        // 0000000017C0: DC6E0000 00000002
	s_nop 0                                                    // 0000000017C8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017CC: BFB60003
	s_endpgm                                                   // 0000000017D0: BFB00000
