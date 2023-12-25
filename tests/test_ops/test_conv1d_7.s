
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_8_6_11>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s10, s15, 11                                     // 000000001714: 960A8B0F
	s_mov_b32 s8, s13                                          // 000000001718: BE88000D
	s_ashr_i32 s11, s10, 31                                    // 00000000171C: 860B9F0A
	s_mov_b32 s2, s15                                          // 000000001720: BE82000F
	s_lshl_b64 s[10:11], s[10:11], 2                           // 000000001724: 848A820A
	v_mov_b32_e32 v1, 0                                        // 000000001728: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	s_add_u32 s3, s6, s10                                      // 000000001730: 80030A06
	s_addc_u32 s10, s7, s11                                    // 000000001734: 820A0B07
	s_ashr_i32 s9, s13, 31                                     // 000000001738: 86099F0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000173C: BF870499
	s_lshl_b64 s[6:7], s[8:9], 2                               // 000000001740: 84868208
	s_add_u32 s8, s3, s6                                       // 000000001744: 80080603
	s_addc_u32 s9, s10, s7                                     // 000000001748: 8209070A
	s_ashr_i32 s15, s14, 31                                    // 00000000174C: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[10:11], s[14:15], 2                           // 000000001754: 848A820E
	s_add_u32 s0, s0, s10                                      // 000000001758: 80000A00
	s_addc_u32 s1, s1, s11                                     // 00000000175C: 82010B01
	s_load_b32 s8, s[8:9], null                                // 000000001760: F4000204 F8000000
	s_load_b32 s9, s[0:1], null                                // 000000001768: F4000240 F8000000
	s_mul_i32 s0, s2, 0x42                                     // 000000001770: 9600FF02 00000042
	s_mul_i32 s2, s14, 11                                      // 000000001778: 96028B0E
	s_ashr_i32 s1, s0, 31                                      // 00000000177C: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001780: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001784: 84808200
	s_add_u32 s4, s4, s0                                       // 000000001788: 80040004
	s_addc_u32 s5, s5, s1                                      // 00000000178C: 82050105
	s_ashr_i32 s3, s2, 31                                      // 000000001790: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001794: BF870499
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001798: 84808202
	s_add_u32 s0, s4, s0                                       // 00000000179C: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017A0: 82010105
	s_add_u32 s0, s0, s6                                       // 0000000017A4: 80000600
	s_addc_u32 s1, s1, s7                                      // 0000000017A8: 82010701
	s_waitcnt lgkmcnt(0)                                       // 0000000017AC: BF89FC07
	v_mul_f32_e64 v0, s8, s9                                   // 0000000017B0: D5080000 00001208
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017B8: BF870001
	v_max_f32_e32 v0, 0, v0                                    // 0000000017BC: 20000080
	global_store_b32 v1, v0, s[0:1]                            // 0000000017C0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017C8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017CC: BFB60003
	s_endpgm                                                   // 0000000017D0: BFB00000
